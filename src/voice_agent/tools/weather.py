"""Current weather from Open-Meteo (free, no API key). Needs internet; offline it says so."""

from __future__ import annotations

from functools import partial

import httpx

from voice_agent.tools.base import Tool, ToolResult

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
CODES = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
         51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
         71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers",
         82: "heavy rain showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def get_weather(tb=None, city: str = "", client: httpx.Client | None = None) -> ToolResult:
    # This function finds the city and returns its current weather.
    city = (city or getattr(tb, "home_city", "") or "").strip()
    if not city:
        return ToolResult(False, error="no city given (set tools.home_city in config.yaml)")
    c = client or httpx.Client(timeout=4)
    try:
        geo = c.get(GEOCODE, params={"name": city, "count": 1}).json().get("results") or []
        if not geo:
            return ToolResult(False, {"city": city}, error=f"city {city!r} not found")
        g = geo[0]
        cur = c.get(FORECAST, params={"latitude": g["latitude"], "longitude": g["longitude"],
                                      "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"}
                    ).json()["current"]
    except (httpx.HTTPError, KeyError, ValueError) as e:
        return ToolResult(False, {"city": city}, error=f"no live weather (offline?): {type(e).__name__}")
    finally:
        if client is None:
            c.close()
    return ToolResult(True, {"city": g.get("name", city), "country": g.get("country", ""),
                             "temperature_c": cur["temperature_2m"], "humidity_percent": cur["relative_humidity_2m"],
                             "wind_kmh": cur["wind_speed_10m"], "condition": CODES.get(cur["weather_code"], "unknown")})


def tools(tb) -> list[Tool]:
    return [Tool("get_weather", "Current weather for a city (needs internet).", {"city": {"type": "string"}},
                 partial(get_weather, tb), timeout_s=10)]
