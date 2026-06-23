from __future__ import annotations

import os
from io import BytesIO
from math import ceil, floor, log, pi, radians, sin
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

from network import CUSTOMERS, LANE_DISTANCE_KM, WAREHOUSES, nearest_warehouse

WIDTH = 960
HEIGHT = 680
TILE_SIZE = 256
ZOOM = 7
TILE_URL_TEMPLATE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "SDA-Simulator-Logistics-Map/0.1"
TILE_CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "sda_simulator"
    / "osm_tiles"
)

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

NODE_COORDS = {
    "W_MADRID": (-3.72, 40.44),
    "W_BARCELONA": (2.16, 41.40),
    "W_VALENCIA": (-0.40, 39.49),
    "C_MADRID_CENTRO": (-3.67, 40.39),
    "C_ALCALA": (-3.36, 40.48),
    "C_BARCELONA_PORT": (2.18, 41.35),
    "C_TARRAGONA": (1.25, 41.12),
    "C_VALENCIA": (-0.33, 39.47),
    "C_CASTELLON": (-0.05, 39.99),
    "C_ZARAGOZA": (-0.89, 41.65),
    "C_BILBAO": (-2.94, 43.26),
    "C_SEVILLA": (-5.98, 37.39),
    "C_MALAGA": (-4.42, 36.72),
    "C_MURCIA": (-1.13, 37.99),
    "C_GIRONA": (2.82, 41.98),
}

NODE_LABELS = {
    "W_MADRID": "Madrid WH",
    "W_BARCELONA": "Barcelona WH",
    "W_VALENCIA": "Valencia WH",
    "C_MADRID_CENTRO": "Madrid",
    "C_ALCALA": "Alcala",
    "C_BARCELONA_PORT": "Barcelona port",
    "C_TARRAGONA": "Tarragona",
    "C_VALENCIA": "Valencia",
    "C_CASTELLON": "Castellon",
    "C_ZARAGOZA": "Zaragoza",
    "C_BILBAO": "Bilbao",
    "C_SEVILLA": "Sevilla",
    "C_MALAGA": "Malaga",
    "C_MURCIA": "Murcia",
    "C_GIRONA": "Girona",
}

LABEL_OFFSETS = {
    "W_MADRID": (-14, -17, "end"),
    "W_BARCELONA": (14, -12, "start"),
    "W_VALENCIA": (12, 23, "start"),
    "C_MADRID_CENTRO": (-12, 18, "end"),
    "C_ALCALA": (11, -8, "start"),
    "C_BARCELONA_PORT": (12, 16, "start"),
    "C_TARRAGONA": (12, 18, "start"),
    "C_VALENCIA": (12, -10, "start"),
    "C_CASTELLON": (12, -10, "start"),
    "C_ZARAGOZA": (12, -8, "start"),
    "C_BILBAO": (-12, -10, "end"),
    "C_SEVILLA": (-12, 22, "end"),
    "C_MALAGA": (10, 22, "start"),
    "C_MURCIA": (12, 18, "start"),
    "C_GIRONA": (12, -8, "start"),
}

WAREHOUSE_COLORS = {
    "W_MADRID": "#d95f02",
    "W_BARCELONA": "#1b9e77",
    "W_VALENCIA": "#386cb0",
}


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def lonlat_to_world_px(lon: float, lat: float) -> tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = TILE_SIZE * (2**ZOOM)
    x = (lon + 180.0) / 360.0 * n
    y = (0.5 - log((1.0 + sin(radians(lat))) / (1.0 - sin(radians(lat)))) / (4 * pi)) * n
    return x, y


def map_bounds() -> tuple[float, float, float, float]:
    longitudes = [lon for lon, _lat in NODE_COORDS.values()]
    latitudes = [lat for _lon, lat in NODE_COORDS.values()]
    min_lon = min(longitudes) - 1.0
    max_lon = max(longitudes) + 1.0
    min_lat = min(latitudes) - 0.8
    max_lat = max(latitudes) + 1.7

    min_x, max_y = lonlat_to_world_px(min_lon, min_lat)
    max_x, min_y = lonlat_to_world_px(max_lon, max_lat)
    box_width = max_x - min_x
    box_height = max_y - min_y
    target_ratio = WIDTH / HEIGHT

    if box_width / box_height > target_ratio:
        target_height = box_width / target_ratio
        delta = (target_height - box_height) / 2
        min_y -= delta
        max_y += delta
    else:
        target_width = box_height * target_ratio
        delta = (target_width - box_width) / 2
        min_x -= delta
        max_x += delta

    return min_x, min_y, max_x, max_y


WORLD_MIN_X, WORLD_MIN_Y, WORLD_MAX_X, WORLD_MAX_Y = map_bounds()
SCALE_X = WIDTH / (WORLD_MAX_X - WORLD_MIN_X)
SCALE_Y = HEIGHT / (WORLD_MAX_Y - WORLD_MIN_Y)


def project(lon: float, lat: float) -> tuple[float, float]:
    world_x, world_y = lonlat_to_world_px(lon, lat)
    x = (world_x - WORLD_MIN_X) * SCALE_X
    y = (world_y - WORLD_MIN_Y) * SCALE_Y
    return x, y


def tile_cache_path(z: int, x: int, y: int) -> Path:
    return TILE_CACHE_DIR / str(z) / str(x) / f"{y}.png"


def read_tile(z: int, x: int, y: int) -> bytes:
    cache_path = tile_cache_path(z, x, y)
    if cache_path.exists():
        return cache_path.read_bytes()

    url = TILE_URL_TEMPLATE.format(z=z, x=x, y=y)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=20) as response:
            data = response.read()
    except URLError as exc:
        raise RuntimeError(f"Could not fetch map tile {url}: {exc}") from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def paste_clipped(base: Image.Image, tile: Image.Image, left: int, top: int) -> None:
    right = left + tile.width
    bottom = top + tile.height
    crop_left = max(0, -left)
    crop_top = max(0, -top)
    crop_right = tile.width - max(0, right - base.width)
    crop_bottom = tile.height - max(0, bottom - base.height)
    if crop_left >= crop_right or crop_top >= crop_bottom:
        return

    base.paste(
        tile.crop((crop_left, crop_top, crop_right, crop_bottom)),
        (left + crop_left, top + crop_top),
    )


def add_tile_background(image: Image.Image) -> None:
    min_tile_x = max(0, floor(WORLD_MIN_X / TILE_SIZE))
    max_tile_x = min((2**ZOOM) - 1, floor(WORLD_MAX_X / TILE_SIZE))
    min_tile_y = max(0, floor(WORLD_MIN_Y / TILE_SIZE))
    max_tile_y = min((2**ZOOM) - 1, floor(WORLD_MAX_Y / TILE_SIZE))

    for tile_y in range(min_tile_y, max_tile_y + 1):
        for tile_x in range(min_tile_x, max_tile_x + 1):
            tile = Image.open(BytesIO(read_tile(ZOOM, tile_x, tile_y))).convert("RGBA")
            x = (tile_x * TILE_SIZE - WORLD_MIN_X) * SCALE_X
            y = (tile_y * TILE_SIZE - WORLD_MIN_Y) * SCALE_Y
            width = ceil(TILE_SIZE * SCALE_X)
            height = ceil(TILE_SIZE * SCALE_Y)
            resized = tile.resize((width, height), Image.Resampling.LANCZOS)
            paste_clipped(image, resized, floor(x), floor(y))

    wash = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 46))
    image.alpha_composite(wash)


def draw_text(
    draw: ImageDraw.ImageDraw,
    label: str,
    x: float,
    y: float,
    *,
    size: int = 12,
    fill: str = "#23313f",
    bold: bool = False,
    anchor: str = "start",
    stroke_width: int = 0,
) -> None:
    anchors = {"start": "lm", "end": "rm", "middle": "mm", "top": "lt"}
    draw.text(
        (round(x), round(y)),
        label,
        font=font(size, bold=bold),
        fill=fill,
        anchor=anchors[anchor],
        stroke_fill="#ffffff",
        stroke_width=stroke_width,
    )


def draw_lanes(draw: ImageDraw.ImageDraw) -> None:
    for highlighted in (False, True):
        for warehouse in WAREHOUSES:
            wx, wy = project(*NODE_COORDS[warehouse])
            color = WAREHOUSE_COLORS[warehouse]

            for customer in CUSTOMERS:
                is_nearest = nearest_warehouse(customer) == warehouse
                if is_nearest != highlighted:
                    continue

                cx, cy = project(*NODE_COORDS[customer])
                points = (round(wx), round(wy), round(cx), round(cy))
                width = 4 if highlighted else 1
                alpha = 220 if highlighted else 72
                draw.line(points, fill=(255, 255, 255, 190), width=width + 3)
                draw.line(points, fill=hex_to_rgba(color, alpha), width=width)

                if highlighted:
                    distance = LANE_DISTANCE_KM[warehouse][customer]
                    mx = wx + (cx - wx) * 0.58
                    my = wy + (cy - wy) * 0.58
                    draw_text(
                        draw,
                        f"{distance} km",
                        mx,
                        my - 4,
                        size=10,
                        fill=color,
                        bold=True,
                        anchor="middle",
                        stroke_width=3,
                    )


def draw_nodes(draw: ImageDraw.ImageDraw) -> None:
    for customer in CUSTOMERS:
        x, y = project(*NODE_COORDS[customer])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#ffffff", outline="#23313f", width=2)
        dx, dy, anchor = LABEL_OFFSETS[customer]
        draw_text(
            draw,
            NODE_LABELS[customer],
            x + dx,
            y + dy,
            size=11,
            anchor=anchor,
            stroke_width=3,
        )

    for warehouse in WAREHOUSES:
        x, y = project(*NODE_COORDS[warehouse])
        color = WAREHOUSE_COLORS[warehouse]
        diamond = [(x, y - 11), (x + 11, y), (x, y + 11), (x - 11, y)]
        draw.polygon(diamond, fill="#ffffff")
        draw.line([*diamond, diamond[0]], fill="#ffffff", width=6, joint="curve")
        draw.polygon(diamond, fill=color)
        draw.line([*diamond, diamond[0]], fill="#1f2933", width=2, joint="curve")
        dx, dy, anchor = LABEL_OFFSETS[warehouse]
        draw_text(
            draw,
            NODE_LABELS[warehouse],
            x + dx,
            y + dy,
            size=12,
            fill="#15202b",
            bold=True,
            anchor=anchor,
            stroke_width=4,
        )


def draw_header(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle((24, 22, 628, 78), radius=6, fill=(255, 255, 255, 230))
    draw_text(
        draw,
        "Spain logistics network",
        42,
        33,
        size=22,
        fill="#15202b",
        bold=True,
        anchor="top",
    )
    draw_text(
        draw,
        "Heavy lines show each customer's nearest warehouse; faint lines are "
        "feasible alternatives.",
        42,
        66,
        size=12,
        fill="#52616f",
        anchor="start",
    )


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    legend_width = 354
    x = WIDTH - legend_width - 40
    y = 82
    draw.rounded_rectangle(
        (x, y, x + legend_width, y + 126),
        radius=6,
        fill=(255, 255, 255, 240),
        outline="#798896",
        width=1,
    )
    draw_text(draw, "Nearest-lane color", x + 16, y + 20, size=12, bold=True, anchor="top")
    draw_text(draw, "Node symbols", x + 204, y + 20, size=12, bold=True, anchor="top")

    for index, warehouse in enumerate(WAREHOUSES):
        row_y = y + 52 + index * 26
        color = WAREHOUSE_COLORS[warehouse]
        draw.line((x + 18, row_y, x + 54, row_y), fill=color, width=4)
        draw_text(draw, NODE_LABELS[warehouse], x + 66, row_y, size=11, anchor="start")

    symbol_x = x + 222
    warehouse_y = y + 56
    customer_y = y + 88
    diamond = [
        (symbol_x, warehouse_y - 9),
        (symbol_x + 9, warehouse_y),
        (symbol_x, warehouse_y + 9),
        (symbol_x - 9, warehouse_y),
    ]
    draw.polygon(diamond, fill="#64748b")
    draw.line([*diamond, diamond[0]], fill="#1f2933", width=2, joint="curve")
    draw_text(draw, "Warehouse", symbol_x + 22, warehouse_y, size=11, anchor="start")

    draw.ellipse(
        (symbol_x - 7, customer_y - 7, symbol_x + 7, customer_y + 7),
        fill="#ffffff",
        outline="#23313f",
        width=2,
    )
    draw_text(draw, "Customer", symbol_x + 22, customer_y, size=11, anchor="start")


def draw_attribution(draw: ImageDraw.ImageDraw) -> None:
    label = "Map data © OpenStreetMap contributors"
    x = WIDTH - 252
    y = HEIGHT - 30
    draw.rounded_rectangle(
        (x - 10, y - 10, x + 224, y + 14),
        radius=4,
        fill=(255, 255, 255, 224),
    )
    draw_text(draw, label, x, y, size=10, fill="#334155", anchor="start")


def hex_to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    color = color.removeprefix("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha


def build_map_image() -> Image.Image:
    missing = sorted(set(WAREHOUSES + CUSTOMERS) - set(NODE_COORDS))
    if missing:
        raise ValueError(f"Missing coordinates for: {', '.join(missing)}")

    image = Image.new("RGBA", (WIDTH, HEIGHT), "#ffffff")
    add_tile_background(image)
    draw = ImageDraw.Draw(image, "RGBA")
    draw_header(draw)
    draw_lanes(draw)
    draw_nodes(draw)
    draw_legend(draw)
    draw_attribution(draw)
    return image


def main() -> None:
    output = Path(__file__).with_name("logistics_network_map.png")
    build_map_image().convert("RGB").save(output, optimize=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
