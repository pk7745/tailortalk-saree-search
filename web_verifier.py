"""
Webpage Verification Module for TailorTalk Saree Search.

Provides secure, SSRF-protected server-side fetching and evidence extraction
from official merchant product web pages.
"""
import ipaddress
import json
import re
import socket
from functools import lru_cache
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import config

# Allowed merchant domain patterns for official product verification
ALLOWED_DOMAINS = [
    "houseofbyrappa.com",
    "byrappa.com",
    "tailortalk.app",
    "streamlit.app",
]

BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.169.254/32"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_safe_url(url: str) -> bool:
    """
    Validates URL safety against Server-Side Request Forgery (SSRF).
    Blocks private IPs, localhost, cloud metadata endpoints, invalid schemes, and non-allowlisted domains.
    """
    if not url or not isinstance(url, str):
        return False

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    hostname_lower = hostname.lower()
    if hostname_lower in ("localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"):
        return False

    # Verify IP address resolution safety
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for family, socktype, proto, canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)

            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
                return False

            for net in BLOCKED_IP_NETWORKS:
                if ip_obj in net:
                    return False
    except Exception:
        # If DNS resolution fails, block the request safely
        return False

    # Domain allowlisting: check if hostname matches known merchant domains
    domain_allowed = any(domain in hostname_lower for domain in ALLOWED_DOMAINS)
    return domain_allowed


def _extract_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extracts Product or Offer structured data from JSON-LD scripts."""
    extracted = {}
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                items = data
            else:
                items = [data]

            for item in items:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("@type", "")).lower()
                if "product" in item_type or "offer" in item_type:
                    if item.get("name"):
                        extracted["product_name"] = str(item["name"])
                    if item.get("description"):
                        extracted["description"] = str(item["description"])
                    if item.get("sku"):
                        extracted["sku"] = str(item["sku"])
                    if item.get("material"):
                        extracted["material"] = str(item["material"])

                    offers = item.get("offers")
                    if isinstance(offers, dict):
                        if offers.get("price"):
                            extracted["price"] = str(offers["price"])
                        if offers.get("priceCurrency"):
                            extracted["currency"] = str(offers["priceCurrency"])
                        if offers.get("availability"):
                            avail = str(offers["availability"])
                            extracted["availability"] = "InStock" if "InStock" in avail else "OutOfStock"
                    elif isinstance(offers, list) and len(offers) > 0 and isinstance(offers[0], dict):
                        if offers[0].get("price"):
                            extracted["price"] = str(offers[0]["price"])
                        if offers[0].get("priceCurrency"):
                            extracted["currency"] = str(offers[0]["priceCurrency"])
                        if offers[0].get("availability"):
                            avail = str(offers[0]["availability"])
                            extracted["availability"] = "InStock" if "InStock" in avail else "OutOfStock"

                    if item.get("price"):
                        extracted["price"] = str(item["price"])
        except Exception:
            continue
    return extracted


def _extract_meta_tags(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extracts product information from OpenGraph and standard HTML meta tags."""
    meta_info = {}

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        meta_info["product_name"] = str(og_title["content"])

    og_price = soup.find("meta", property="og:price:amount") or soup.find("meta", property="product:price:amount")
    if og_price and og_price.get("content"):
        meta_info["price"] = str(og_price["content"])

    og_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
    if og_desc and og_desc.get("content"):
        meta_info["description"] = str(og_desc["content"])

    return meta_info


@lru_cache(maxsize=128)
def fetch_official_product_details(url: str) -> Dict[str, Any]:
    """
    Fetches and extracts structured evidence from an official product web page.
    Includes SSRF security safeguards, JSON-LD parsing, meta tag extraction, and LRU caching.
    """
    if not url or not isinstance(url, str):
        return {
            "success": False,
            "error": "Invalid or empty product URL provided.",
            "evidence_source": "official_product_page",
        }

    url = url.strip()

    if not _is_safe_url(url):
        return {
            "success": False,
            "error": f"URL '{url}' failed security validation (SSRF/domain restriction).",
            "evidence_source": "official_product_page",
        }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5, stream=True)
        response.raise_for_status()

        # Enforce maximum response size limit of 1MB
        content_bytes = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            content_bytes.extend(chunk)
            if len(content_bytes) > 1024 * 1024:
                break

        html_content = content_bytes.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "html.parser")

        json_ld_data = _extract_json_ld(soup)
        meta_data = _extract_meta_tags(soup)

        # Merge extracted evidence, prioritizing JSON-LD
        product_name = json_ld_data.get("product_name") or meta_data.get("product_name") or soup.title.string if soup.title else ""
        price = json_ld_data.get("price") or meta_data.get("price") or ""
        currency = json_ld_data.get("currency") or "INR"
        availability = json_ld_data.get("availability") or "InStock"
        description = json_ld_data.get("description") or meta_data.get("description") or ""

        # Extract text snippets for specs (fabric, color, blouse, care)
        text_snippet = soup.get_text(separator=" ", strip=True)[:1500]

        fabric_match = re.search(r"\b(banarasi|organza|tussar|linen|satin|munga|silk|cotton|georgette|chiffon)\b", text_snippet, re.IGNORECASE)
        fabric = fabric_match.group(0).title() if fabric_match else json_ld_data.get("material", "Silk Blend")

        return {
            "success": True,
            "evidence_source": "official_product_page",
            "product_url": url,
            "product_name": str(product_name).strip(),
            "price": str(price).strip(),
            "currency": str(currency).strip(),
            "availability": str(availability).strip(),
            "fabric": str(fabric).strip(),
            "description": str(description).strip()[:300],
            "raw_text_summary": text_snippet[:500],
        }

    except Exception as ex:
        return {
            "success": False,
            "error": f"Failed to fetch official product webpage: {ex}",
            "evidence_source": "official_product_page",
            "product_url": url,
        }
