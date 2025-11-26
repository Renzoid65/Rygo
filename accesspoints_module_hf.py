import gradio as gr
import psycopg2
import pandas as pd
import random
import re
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import PIL
from math import sqrt
import pickle
from cryptography.fernet import Fernet, InvalidToken
import ast
from typing import Optional, Dict, Any
import time
import json
import os


# ===== Optional QR: try qrcode, else Pillow fallback (no crash on Spaces) =====
try:
    import qrcode  # external; may not exist on Spaces
    def _make_qr_image(data: str, box_size: int = 20, border: int = 2) -> Image.Image:
        qr = qrcode.QRCode(version=1, box_size=box_size, border=border)
        qr.add_data(data)
        qr.make(fit=True)
        return qr.make_image(fill_color="#000000", back_color="#FFFFFF").convert("RGB")
except Exception:
    # Fallback: draw a simple placeholder (NOT a real QR) so app keeps working
    def _make_qr_image(data: str, box_size: int = 20, border: int = 2) -> Image.Image:
        w = h = 400
        img = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w - 1, h - 1], outline="black")
        msg = "QR lib missing"
        d.text((12, 12), msg, fill="black")
        d.text((12, 40), str(data), fill="black")
        return img


# ===== retrieve params=====
APP_DIR = Path(__file__).parent

def decrypt_nhost() -> dict:
    key = os.getenv("SECRET_KEY", "").strip()
    if not key:
        raise RuntimeError("SECRET_KEY not set in this Space (Settings → Secrets).")

    enc_path = APP_DIR / "nhost_params.enc"
    if not enc_path.exists():
        raise RuntimeError("Encrypted DB config (nhost_params.enc) not found in repo.")

    token = enc_path.read_bytes()
    decrypted = Fernet(key.encode("utf-8")).decrypt(token).decode("utf-8")

    # Prefer JSON; allow Python-literal fallback
    try:
        params = json.loads(decrypted)
    except json.JSONDecodeError:
        params = ast.literal_eval(decrypted)  # handles "{'host': ...}" format

    return params


def get_connection():
    params = decrypt_nhost()
    return psycopg2.connect(params)



# ─────────────────────────────────────────────
# Active User: same approach as property/permissions modules
# ─────────────────────────────────────────────

NEW_DATA_FILE = Path(__file__).parent / "openqr_active_user.pkl.enc"
NEW_KEY_FILE  = Path(__file__).parent / "openqr_secret.key"
LEGACY_DATA_FILE = Path(__file__).parent / "oqrdata.pkl.enc"
LEGACY_KEY_FILE  = Path(__file__).parent / "oqrdata.key"

def _read_encrypted_pickle(data_path: Path, key_path: Path):
    try:
        if not data_path.exists() or not key_path.exists():
            return None
        key = key_path.read_bytes()
        f = Fernet(key)
        blob = data_path.read_bytes()
        raw = f.decrypt(blob)
        return pickle.loads(raw)
    except (InvalidToken, Exception):
        return None


ActiveUserID: Optional[int] = None


def _coerce_uid(val) -> Optional[int]:
    try:
        iv = int(str(val).strip())
        return iv if iv > 0 else None
    except Exception:
        return None
    
def resolve_and_cache_uid(get_user_id=None) -> Optional[int]:
    """
    Always resolve the *current* ActiveUserID from:
      1) The provided callback (get_user_id)
      2) The OPENQR_ACTIVE_USER_ID / OQR_ACTIVE_USER_ID env var
      3) The encrypted pickle
    and then update the global ActiveUserID.

    This avoids stale cached values when the app switches between
    different manager / installer users in the same Python process.
    """
    global ActiveUserID

    uid: Optional[int] = None

    # 1) callback first (this is the authoritative source from app.py)
    if callable(get_user_id):
        try:
            uid = _coerce_uid(get_user_id())
        except Exception:
            uid = None

    # 2) env next
    if not uid:
        env_val = _coerce_uid(
            os.getenv("OPENQR_ACTIVE_USER_ID") or os.getenv("OQR_ACTIVE_USER_ID")
        )
        if env_val:
            uid = env_val

    # 3) encrypted pickle last (for legacy/standalone launches)
    if not uid:
        info = _read_encrypted_pickle(NEW_DATA_FILE, NEW_KEY_FILE) or _read_encrypted_pickle(
            LEGACY_DATA_FILE, LEGACY_KEY_FILE
        )
        if info:
            for k in ("ActiveUserID", "UserID", "MUserID"):
                v = _coerce_uid(info.get(k))
                if v:
                    uid = v
                    break

    # Update global cache with the *current* resolved UID (may be None)
    ActiveUserID = uid or None
    return ActiveUserID

APGROUP_WARNING_CSS = """
#apgroup-warning-box,
#apgroup-warning-box * {
    background-color: white !important;
    color: black !important;
}

#apgroup-warning-box .prose,
#apgroup-warning-box .markdown,
#apgroup-warning-box pre,
#apgroup-warning-box code {
    background-color: white !important;
}
"""


def launch_accesspoints_module(get_user_id=None):

    if _coerce_uid(ActiveUserID):
        return ActiveUserID

    # ========== UTILS ==========
    def generate_id():
        return random.randint(111111111111111, 999999999999999)

    def _get_installers_flag(user_id: int | None) -> bool:
        if not user_id:
            return False
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute('SELECT "Installers" FROM managerusers WHERE "MUserID" = %s', (int(user_id),))
            row = cur.fetchone()
            cur.close()
            conn.close()
            return bool(row[0]) if row and row[0] is not None else False
        except Exception as e:
            print(f"⚠️ Installers check failed: {e}")
            return False

    def _get_property_name(prop_id: int | str, user_id: int | None) -> str:
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                'SELECT "Name" FROM property WHERE "PropID" = %s AND "UserID" = %s AND "Active" = TRUE',
                (int(prop_id), int(user_id) if user_id is not None else None),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            return row[0] if row and row[0] is not None else ""
        except Exception as e:
            print(f"⚠️ _get_property_name failed: {e}")
            return ""

    def _get_font(size: int, font_path: str | None = None) -> ImageFont.FreeTypeFont:
        candidates: list[str | Path] = []
        if font_path:
            candidates.append(font_path)
        pil_fonts = Path(PIL.__file__).parent / "fonts"
        candidates += [pil_fonts / "DejaVuSans.ttf", pil_fonts / "DejaVuSans-Bold.ttf"]
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
        for p in candidates:
            try:
                p = Path(p)
                if p.exists():
                    return ImageFont.truetype(str(p), size)
            except Exception:
                pass
        return ImageFont.load_default()

    # ---------- Geometry helper (flat-top regular hex that fits a bbox) ----------
    def hex_points_flat_top(width: int, height: int):
        A = height / 2.0  # apothem
        W_expected = (4.0 * A) / sqrt(3)
        scale_x = width / W_expected
        top_half = (A / sqrt(3)) * scale_x
        cx, cy = width / 2.0, height / 2.0
        pts = [
            (cx - top_half, 0),          # top-left
            (cx + top_half, 0),          # top-right
            (width, cy),                 # right-mid
            (cx + top_half, height),     # bottom-right
            (cx - top_half, height),     # bottom-left
            (0, cy),                     # left-mid
        ]
        return [(int(round(x)), int(round(y))) for (x, y) in pts]

    def create_save_png_qrcode(
        data: str,
        *,
        border_thickness: int = 60,
        border_color: str = "#EF5635",
        top_text: str = "RyGo QR",
        bottom_text: str = "Scan to open",
        top_font_size: int = 50,
        bottom_font_size: int = 50,
        text_color: str = "#000000",
        font_path: str | None = None,
        box_size: int = 20,
        qr_border_modules: int = 2
    ) -> bytes:
        if text_color is None:
            text_color = border_color

        # Use optional qrcode lib if available, else fallback placeholder
        qr_img = _make_qr_image(str(data), box_size=box_size, border=qr_border_modules)
        qr_w, qr_h = qr_img.size

        font_top = _get_font(top_font_size, font_path)
        font_bottom = _get_font(bottom_font_size, font_path)
        tmp = Image.new("RGB", (10, 10))
        dtmp = ImageDraw.Draw(tmp)

        def tsize(s, f):
            if not s:
                return (0, 0)
            l, t, r, b = dtmp.textbbox((0, 0), s, font=f)
            return (r - l, b - t)

        top_w, top_h = tsize(top_text, font_top)
        bot_w, bot_h = tsize(bottom_text, font_bottom)

        pad_top = max(8, int(top_font_size * 0.35))
        pad_bot = max(8, int(bottom_font_size * 0.35))
        top_band_h = (top_h + 2 * pad_top) if top_text else 0
        bottom_band_h = (bot_h + 2 * pad_bot) if bottom_text else 0

        inner_content_w = max(qr_w, top_w, bot_w)
        inner_content_h = top_band_h + qr_h + bottom_band_h

        need_top = (max(top_w, bot_w)) * (sqrt(3) / 2.0)
        need_center = inner_content_w * (sqrt(3) / 4.0)
        need_height = inner_content_h / 2.0
        A_in = int(round(max(need_top, need_center, need_height)))

        inner_w = int(round((4.0 * A_in) / sqrt(3)))
        inner_h = int(round(2.0 * A_in))

        A_out = A_in + border_thickness
        outer_w = int(round((4.0 * A_out) / sqrt(3)))
        outer_h = int(round(2.0 * A_out))

        canvas = Image.new("RGB", (outer_w, outer_h), "white")
        draw = ImageDraw.Draw(canvas)
        outer_hex = hex_points_flat_top(outer_w, outer_h)
        draw.polygon(outer_hex, fill=border_color)

        inner_offset = ((outer_w - inner_w) // 2, (outer_h - inner_h) // 2)
        inner_hex_abs = [(x + inner_offset[0], y + inner_offset[1]) for x, y in hex_points_flat_top(inner_w, inner_h)]
        draw.polygon(inner_hex_abs, fill="white")

        inner_img = Image.new("RGB", (inner_w, inner_h), "white")
        idraw = ImageDraw.Draw(inner_img)

        if top_text:
            tx = (inner_w - top_w) // 2
            ty = (top_band_h - top_h) // 2
            idraw.text((tx, ty), top_text, font=font_top, fill=text_color)

        qr_x = (inner_w - qr_w) // 2
        qr_y = top_band_h
        inner_img.paste(qr_img, (qr_x, qr_y))

        if bottom_text:
            bx = (inner_w - bot_w) // 2
            by = top_band_h + qr_h + (bottom_band_h - bot_h) // 2
            idraw.text((bx, by), bottom_text, font=font_bottom, fill=text_color)

        mask_local = Image.new("L", (inner_w, inner_h), 0)
        mdraw = ImageDraw.Draw(mask_local)
        mdraw.polygon(hex_points_flat_top(inner_w, inner_h), fill=255)
        canvas.paste(inner_img, inner_offset, mask_local)

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return buf.getvalue()

    # ---------- NEW: helpers for time handling ----------
    _TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

    def _valid_hhmm(s: str) -> bool:
        if not s:
            return False
        return bool(_TIME_RE.match(s.strip()))

    def _format_time_for_ui(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        # If it's a datetime.time or similar, use strftime
        if hasattr(val, "strftime"):
            return val.strftime("%H:%M")
        s = str(val).strip()
        # Handle strings like "HH:MM:SS" or "HH:MM"
        if len(s) >= 5 and s[2] == ":":
            return s[:5]
        return s


    # ========== MAIN MODULE ==========
    
    with gr.Blocks() as demo:
        
        # Inject APGROUP_WARNING_CSS via a <style> tag
        gr.HTML(f"<style>{APGROUP_WARNING_CSS}</style>")
        
        with gr.Tab("Manage Access Points"):

            # ---- property options for current user ----

            def load_installer_options():
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(
                    'SELECT "MUserName" '
                    'FROM managerusers '
                    'WHERE "Installers" = TRUE '
                    'ORDER BY "MUserName" ASC'
                )
                rows = cur.fetchall()
                cur.close()
                conn.close()

                # First option = sentinel
                opts: list[str] = ["<to be appointed>"]

                # Each row is a 1-tuple: (MUserName,)
                for (cname,) in rows:
                    if cname is None:
                        continue
                    # Only show the installer name, no IDs
                    opts.append(str(cname))

                return opts

            def load_property_options():
                uid = _uid()
                print(f"DEBUG load_property(): ActiveUserID={uid}")
                if uid is None:
                    return []
            
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT "PropID", "Name", "Owner" '
                    'FROM property WHERE "UserID" = %s AND "Active" = TRUE',
                    (uid,)
                )
                rows = cursor.fetchall()
                conn.close()
            
                seen = set()
                options = []
                for prop_id, name, owner in rows:
                    key = (prop_id, name, owner)
                    if key in seen:
                        continue
                    seen.add(key)
            
                    # Show ONLY "Name (Owner)" in the dropdown label (no PropID)
                    label = f"{name} ({owner})"
            
                    # Keep PropID in the hidden value so downstream code still works
                    value = f"{prop_id}|{name}|{owner}"
                    options.append((label, value))
            
                return options

            def _uid() -> Optional[int]:
                # convenience wrapper used throughout the module
                return resolve_and_cache_uid(get_user_id)



            def _prime_uid_then_fill_properties():
                # allow time for env/callback to be ready (HF cold start)
                deadline = time.time() + 10.0
                while resolve_and_cache_uid(get_user_id) is None and time.time() < deadline:
                    time.sleep(0.1)

                uid = _uid()
                print("DEBUG (prime): starting with UID =", uid)

                if uid is None:
                    return gr.update(choices=[], value=None, interactive=False)

                # Use options list [(label, value), ...]; build prop_map (label -> (PropID, Name, Owner))
                options = load_property_options()
                labels = [lbl for (lbl, _val) in options]

                prop_map.clear()
                for lbl, val in options:
                    try:
                        pid, pname, owner = val.split("|", 2)
                    except ValueError:
                        pid, pname, owner = "", "", ""
                    prop_map[lbl] = (int(pid) if pid else None, pname, owner)

                return gr.update(choices=labels, value=None, interactive=bool(labels))

            def _apgroup_token_contains(text: str | None, group_name: str) -> bool:
                """
                Check if group_name appears as a full token in a comma-separated list
                (case-insensitive).
                """
                if not text or not group_name:
                    return False
                tokens = [t.strip() for t in str(text).split(",") if t.strip()]
                g = group_name.strip().lower()
                return any(t.lower() == g for t in tokens)

            def _check_apgroup_usage(group_name: str, prop_id):
                """
                Return (in_use, count_parking, count_tenantuser) for this AP group.
                in_use is True if the group_name is found in any BayAccessNo or PedestrianAPs
                for this user and property.
                """
                group_name = (group_name or "").strip()
                if not group_name or not prop_id:
                    return False, 0, 0

                uid = _uid()
                if uid is None:
                    return False, 0, 0

                count_pb = 0
                count_tu = 0
                conn = None
                try:
                    conn = get_connection()
                    cur = conn.cursor()

                    # parkingbays: BayAccessNo
                    cur.execute(
                        'SELECT "BayAccessNo" FROM parkingbays '
                        'WHERE "UserID" = %s AND "PropID" = %s '
                        'AND "BayAccessNo" IS NOT NULL AND "BayAccessNo" <> \'\'',
                        (uid, int(prop_id)),
                    )
                    for (val,) in cur.fetchall():
                        if _apgroup_token_contains(val, group_name):
                            count_pb += 1

                    # tenantuser: PedestrianAPs
                    cur.execute(
                        'SELECT "PedestrianAPs" FROM tenantuser '
                        'WHERE "UserID" = %s AND "PropID" = %s '
                        'AND "PedestrianAPs" IS NOT NULL AND "PedestrianAPs" <> \'\'',
                        (uid, int(prop_id)),
                    )
                    for (val,) in cur.fetchall():
                        if _apgroup_token_contains(val, group_name):
                            count_tu += 1

                except Exception as e:
                    print(f"⚠️ _check_apgroup_usage failed for group={group_name}, PropID={prop_id}: {e}")
                finally:
                    try:
                        if conn is not None:
                            conn.close()
                    except Exception:
                        pass

                in_use = (count_pb > 0 or count_tu > 0)
                return in_use, count_pb, count_tu

            def _remove_group_from_csv(text: str | None, group_name: str) -> str:
                """
                Remove group_name from a comma-separated list of group names, return cleaned string.
                """
                if not text:
                    return ""
                tokens = [t.strip() for t in str(text).split(",") if t.strip()]
                g = group_name.strip().lower()
                new_tokens = [t for t in tokens if t.lower() != g]
                return ", ".join(new_tokens)

            def _cleanup_apgroup_references(group_name: str, prop_id):
                """
                After confirmed delete of AP group:
                  - Remove group_name from parkingbays.BayAccessNo
                  - Remove group_name from tenantuser.PedestrianAPs
                for the current user and the given property.
                """
                group_name = (group_name or "").strip()
                if not group_name or not prop_id:
                    return

                uid = _uid()
                if uid is None:
                    return

                conn = None
                try:
                    conn = get_connection()
                    cur = conn.cursor()

                    # parkingbays cleanup
                    cur.execute(
                        'SELECT "ID","BayAccessNo" FROM parkingbays '
                        'WHERE "UserID" = %s AND "PropID" = %s '
                        'AND "BayAccessNo" IS NOT NULL AND "BayAccessNo" <> \'\'',
                        (uid, int(prop_id)),
                    )
                    rows = cur.fetchall()
                    for row_id, val in rows:
                        new_val = _remove_group_from_csv(val, group_name)
                        if new_val != (val or ""):
                            cur.execute(
                                'UPDATE parkingbays SET "BayAccessNo" = %s WHERE "ID" = %s',
                                (new_val or None, row_id),
                            )

                    # tenantuser cleanup
                    cur.execute(
                        'SELECT "ID","PedestrianAPs" FROM tenantuser '
                        'WHERE "UserID" = %s AND "PropID" = %s '
                        'AND "PedestrianAPs" IS NOT NULL AND "PedestrianAPs" <> \'\'',
                        (uid, int(prop_id)),
                    )
                    rows = cur.fetchall()
                    for row_id, val in rows:
                        new_val = _remove_group_from_csv(val, group_name)
                        if new_val != (val or ""):
                            cur.execute(
                                'UPDATE tenantuser SET "PedestrianAPs" = %s WHERE "ID" = %s',
                                (new_val or None, row_id),
                            )

                    conn.commit()
                    print(f"[DEBUG] Cleaned AP group '{group_name}' references for PropID={prop_id}")

                except Exception as e:
                    if conn is not None:
                        conn.rollback()
                    print(f"❌ _cleanup_apgroup_references failed for group={group_name}, PropID={prop_id}: {e}")
                finally:
                    try:
                        if conn is not None:
                            conn.close()
                    except Exception:
                        pass


            def sync_accesspoints(prop_id):
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT "ID","PropID","NameOfAccessPoint","AccessPointID","UserID",'
                    '       "RestrictedAP","APDeviceName","APInstaller","APInstallerID",'
                    '       "ApInOut","APOpen","APClose","APActive" '
                    'FROM accesspoint '
                    'WHERE "UserID" = %s AND "PropID" = %s '
                    'ORDER BY lower("NameOfAccessPoint") ASC',
                    (_uid(), int(prop_id))
                )
                rows = cursor.fetchall()
                conn.close()
            
                df = pd.DataFrame(
                    rows,
                    columns=[
                        "ID", "PropID", "NameOfAccessPoint", "AccessPointID", "UserID",
                        "RestrictedAP", "APDeviceName", "APInstaller", "APInstallerID",
                        "ApInOut", "APOpen", "APClose", "APActive",
                    ],
                )

                if not df.empty:
                    df["AccessPointID"] = df["AccessPointID"].astype(str).str.replace(r"\.0$", "", regex=True)

                    # format time columns for UI
                    if "APOpen" in df.columns:
                        df["APOpen"] = df["APOpen"].apply(_format_time_for_ui)
                    if "APClose" in df.columns:
                        df["APClose"] = df["APClose"].apply(_format_time_for_ui)

                # hide ID/PropID/UserID/APInstallerID from the table
                df_display = df.drop(columns=["ID", "PropID", "UserID", "APInstallerID", "AccessPointID"])

                if not df_display.empty:
                    # keep main columns at the front
                    desired_front = [
                        c for c in ["NameOfAccessPoint", "ApInOut"]
                        if c in df_display.columns
                    ]
                    others = [c for c in df_display.columns if c not in desired_front]
                    df_display = df_display[desired_front + others]

                    # rename columns for UI (keeping your existing labels)
                    rename_map = {
                        "NameOfAccessPoint": "Access Point",
                        "RestrictedAP": "Restricted",
                        "APDeviceName": "Device Name",
                        "APInstaller": "Installer",
                        "ApInOut": "Direction",
                        "APOpen": "Open ",
                        "APClose": "Close",
                        "APActive": "Active",
                    }
                    df_display = df_display.rename(columns=rename_map)

                return df, df_display


            def get_row_options(df):
                if df is None or df.empty:
                    return []
                return [
                    (f"{row['NameOfAccessPoint']} - {row.get('ApInOut','')}", str(int(row["ID"])))
                    for _, row in df.iterrows()
                ]

            def select_property(label):
                if not label or label not in prop_map:
                    return (
                        gr.update(visible=False),  # accesspoint_table
                        None, None,
                        gr.update(choices=[], value=None, interactive=False, visible=True),
                        gr.update(choices=[], value=None, interactive=False, visible=True),
                        gr.update(visible=False),  # apply_edit_btn
                        gr.update(visible=False, interactive=False),  # delete_btn
                        gr.update(visible=True, interactive=False),   # add_btn visible but inactive
                        gr.update(visible=False),  # edit_group
                        gr.update(value="", visible=False),  # error_msg
                        gr.update(visible=False),  # cancel_edit_btn
                        gr.update(visible=False),  # cancel_delete_btn
                    )
                prop_id, prop_name, owner = prop_map[label]

                # Sync APProperty to property.Name
                try:
                    conn_sync = get_connection()
                    cur_sync = conn_sync.cursor()
                    cur_sync.execute(
                        'UPDATE accesspoint SET "APProperty" = %s '
                        'WHERE "UserID" = %s AND "PropID" = %s '
                        '  AND ( "APProperty" IS NULL OR btrim("APProperty") <> btrim(%s) )',
                        (prop_name, _uid(), int(prop_id), prop_name)
                    )
                    if cur_sync.rowcount:
                        print(f"[DEBUG] Synced accesspoint.APProperty rows: {cur_sync.rowcount} for PropID={prop_id}")
                    conn_sync.commit()
                except Exception as e:
                    try: conn_sync.rollback()
                    except: pass
                    print(f"❌ APProperty sync failed for PropID={prop_id}: {e}")
                finally:
                    try:
                        cur_sync.close(); conn_sync.close()
                    except: pass

                df_full, df_display = sync_accesspoints(prop_id)
                options = get_row_options(df_full)

                return (
                    gr.update(visible=True, value=df_display),
                    df_full,
                    prop_id,
                    gr.update(choices=options, value=None, interactive=not df_full.empty, visible=True),
                    gr.update(choices=options, value=None, interactive=not df_full.empty, visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False, interactive=False),
                    gr.update(visible=True, interactive=True),  # add_btn visible + active
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            def select_row(row_id, df):
                if not row_id or df is None or df.empty:
                    return (
                        "", "", False, "", None,
                        gr.update(choices=load_installer_options(), value=None),
                        "",
                        gr.update(visible=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(interactive=True),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                        gr.update(value=None, visible=False),
                        "",
                        "",
                        False,
                    )
            
                try:
                    selected_id = int(str(row_id))
                    match = df[df["ID"] == selected_id]
                except Exception:
                    match = pd.DataFrame()
            
                if match.empty:
                    return (
                        "", "", False, "", None,
                        gr.update(choices=load_installer_options(), value=None),
                        "",
                        gr.update(visible=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(interactive=True),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                        gr.update(value=None, visible=False),
                        "",
                        "",
                        False,
                    )
            
                row = match.iloc[0]
                opts = load_installer_options()
            
                # Dropdown now shows ONLY names; use saved name directly
                saved_installer_name = (row.get("APInstaller") or "").strip()
                dd_value = saved_installer_name if saved_installer_name in opts else "<to be appointed>"
            
                # Hidden ID textbox gets the saved APInstallerID (if any)
                ap_installer_id = row.get("APInstallerID")
                id_str = str(int(ap_installer_id)) if pd.notna(ap_installer_id) and ap_installer_id is not None else ""
            
                inout_val = row.get("ApInOut")
                inout_sel = (
                    str(inout_val) if isinstance(inout_val, str) and inout_val in ("In", "Out", "N/A") else None
                )
            
                access_point_id = row.get("AccessPointID")
                pngfile = None
                if access_point_id is not None:
                    pngfile = f"{access_point_id}.png"
                    try:
                        bdata = create_save_png_qrcode(str(access_point_id))
                        with open(pngfile, "wb") as f:
                            f.write(bdata)
                        print(f"QR image created in select_row as {pngfile}")
                    except Exception as e:
                        print(f"⚠️ Failed to create QR in select_row: {e}")
                        pngfile = None
                qr_update = gr.update(value=pngfile, visible=True) if pngfile else gr.update(value=None, visible=False)
            
                ap_open_str = _format_time_for_ui(row.get("APOpen"))
                ap_close_str = _format_time_for_ui(row.get("APClose"))
                ap_active_val = bool(row.get("APActive")) if "APActive" in row else False
            
                return (
                    row["NameOfAccessPoint"],
                    gr.update(value=str(row["AccessPointID"]), interactive=False),
                    row["RestrictedAP"],
                    row["APDeviceName"],
                    inout_sel,
                    gr.update(choices=opts, value=dd_value),   # name only
                    id_str,                                    # hidden ID textbox
                    gr.update(visible=True),
                    gr.update(visible=True, interactive=True),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(visible=True),
                    qr_update,
                    ap_open_str,
                    ap_close_str,
                    ap_active_val,
                )


            def installer_changed(dd_value):
                """
                dd_value is the installer NAME or the sentinel "<to be appointed>".
                Return the MUserID as string for the hidden edit_installer_id textbox,
                or "" if none.
                """
                name = (dd_value or "").strip()
                if not name or name == "<to be appointed>":
                    return ""
                try:
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute(
                        'SELECT "MUserID" FROM managerusers WHERE "Installers" = TRUE AND "MUserName" = %s LIMIT 1',
                        (name,)
                    )
                    row = cur.fetchone()
                except Exception as e:
                    print(f"⚠️ installer_changed lookup failed: {e}")
                    row = None
                finally:
                    try:
                        cur.close(); conn.close()
                    except Exception:
                        pass
            
                return str(int(row[0])) if row and row[0] is not None else ""


            def apply_edit(
                selected_row, df, new_name, new_phone, new_restrict, new_device,
                new_inout, dd_installer, ro_installer_id, prop_id,
                new_ap_open, new_ap_close, new_ap_active,
            ):
                if not selected_row or df is None or getattr(df, "empty", True):
                    return (
                        gr.update(), gr.update(), gr.update(),
                        "", "", False, "", None,
                        gr.update(choices=load_installer_options(), value=None),
                        "",
                        gr.update(visible=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(interactive=True),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                        gr.update(value=None, visible=False),
                        "",
                        "",
                        False,
                    )
            
                row_id = None
                sval = str(selected_row).strip()
                try:
                    row_id = int(sval)
                except Exception:
                    m = re.search(r"\((\d+)\)$", sval)
                    if m:
                        row_id = int(m.group(1))
                if row_id is None:
                    return (
                        gr.update(), gr.update(), gr.update(),
                        "", "", False, "", None,
                        gr.update(choices=load_installer_options(), value=None),
                        "",
                        gr.update(visible=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(visible=False, interactive=False),
                        gr.update(interactive=True),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                        gr.update(value=None, visible=False),
                        "",
                        "",
                        False,
                    )
            
                # Resolve installer name + id
                sentinel = "<to be appointed>"
                sel_company = "" if not dd_installer or dd_installer.strip() == sentinel else dd_installer.strip()
            
                # Prefer the hidden textbox ID; if empty, try to resolve by name (fallback)
                sel_company_id = None
                s_id = (ro_installer_id or "").strip()
                if s_id.isdigit():
                    sel_company_id = int(s_id)
                elif sel_company:
                    try:
                        conn = get_connection()
                        cur = conn.cursor()
                        cur.execute(
                            'SELECT "MUserID" FROM managerusers WHERE "Installers" = TRUE AND "MUserName" = %s LIMIT 1',
                            (sel_company,)
                        )
                        row = cur.fetchone()
                        sel_company_id = int(row[0]) if row and row[0] is not None else None
                    except Exception as e:
                        print(f"⚠️ apply_edit lookup failed: {e}")
                        sel_company_id = None
                    finally:
                        try:
                            cur.close(); conn.close()
                        except Exception:
                            pass
            
                # Device blank if no installer picked (keeping your prior behavior)
                if not sel_company:
                    new_device = ""
            
                inout_value = new_inout if new_inout in ("In", "Out", "N/A") else None
            
                # Validate HH:MM
                open_str = (new_ap_open or "").strip()
                close_str = (new_ap_close or "").strip()
                if open_str and not _valid_hhmm(open_str):
                    return (
                        gr.update(),
                        gr.update(value=selected_row),
                        gr.update(),
                        new_name, new_phone, new_restrict,
                        new_device, inout_value,
                        gr.update(value=dd_installer),
                        ro_installer_id,
                        gr.update(visible=True),
                        gr.update(value="⚠️ APOpen must be in HH:MM (24h) format.", visible=True),
                        gr.update(visible=True, interactive=True),
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(visible=True),
                        gr.update(),
                        open_str,
                        close_str,
                        bool(new_ap_active),
                    )
                if close_str and not _valid_hhmm(close_str):
                    return (
                        gr.update(),
                        gr.update(value=selected_row),
                        gr.update(),
                        new_name, new_phone, new_restrict,
                        new_device, inout_value,
                        gr.update(value=dd_installer),
                        ro_installer_id,
                        gr.update(visible=True),
                        gr.update(value="⚠️ APClose must be in HH:MM (24h) format.", visible=True),
                        gr.update(visible=True, interactive=True),
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(visible=True),
                        gr.update(),
                        open_str,
                        close_str,
                        bool(new_ap_active),
                    )
            
                open_val = open_str or None
                close_val = close_str or None
                active_val = bool(new_ap_active)
            
                # --- Update DB (now persists installer name + id) ---
                conn = get_connection()
                cur = conn.cursor()
                try:
                    cur.execute(
                        'UPDATE accesspoint '
                        'SET "NameOfAccessPoint"=%s, '
                        '    "RestrictedAP"=%s, '
                        '    "APDeviceName"=%s, '
                        '    "ApInOut"=COALESCE(%s, "ApInOut"), '
                        '    "APInstaller"=%s, '
                        '    "APInstallerID"=%s, '
                        '    "APOpen"=%s, '
                        '    "APClose"=%s, '
                        '    "APActive"=%s '
                        'WHERE "ID"=%s',
                        (new_name, new_restrict, new_device, inout_value,
                         sel_company, sel_company_id, open_val, close_val, active_val, row_id)
                    )
                    conn.commit()
                finally:
                    cur.close()
                    conn.close()
            
                # --- Refresh UI as before ---
                df_updated, df_display = sync_accesspoints(prop_id)
                options = get_row_options(df_updated)
            
                return (
                    gr.update(value=df_display),
                    gr.update(choices=options, value=None),
                    gr.update(choices=options, value=None, interactive=not df_updated.empty),
                    "", "", False, "", None,
                    gr.update(choices=load_installer_options(), value=None),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False, interactive=False),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    gr.update(visible=False),
                    gr.update(value=None, visible=False),
                    "",
                    "",
                    False,
                )


            def cancel_edit():
                return (
                    "", "", False, "", None,
                    gr.update(choices=load_installer_options(), value=None),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False, interactive=False),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    gr.update(visible=False),
                    "",   # edit_ap_open
                    "",   # edit_ap_close
                    False, # edit_ap_active
                )

            def cancel_delete():
                return gr.update(value=None), gr.update(interactive=False, visible=False), gr.update(interactive=True), gr.update(visible=False)

            def add_row(prop_id):
                if not prop_id:
                    return pd.DataFrame(), gr.update(), gr.update(), gr.update(), None, None
                new_id = generate_id()
                new_access_point_id = generate_id()
                name = "<enter access point name>"

                ap_property_name = _get_property_name(prop_id, _uid())
                ap_installer_value = ""
                ap_installer_id = None

                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO accesspoint '
                    '("ID","PropID","NameOfAccessPoint","AccessPointID","UserID",'
                    ' "RestrictedAP","APDeviceName","APInstaller","APInstallerID","APProperty",'
                    ' "APOpen","APClose","APActive") '
                    'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                    (new_id, int(prop_id), name, int(new_access_point_id), _uid(),
                     False, "", ap_installer_value, ap_installer_id, ap_property_name,
                     None, None, True)  # NEW: APActive defaults TRUE
                )
                conn.commit()
                conn.close()

                # Generate & save PNG (real QR if lib present; placeholder otherwise)
                df_updated, df_display = sync_accesspoints(prop_id)
                options = get_row_options(df_updated)
                return (
                    df_display,                            # 1) df_display
                    gr.update(choices=options, value=None),  # 2) select row to edit
                    gr.update(choices=options, value=None),  # 3) select row to delete
                    gr.update(choices=options, value=None),  # 4) report selector (if any)
                    df_updated,                            # 5) df_updated (state)
                )

            def enable_delete_button(row_id):
                return gr.update(interactive=bool(row_id), visible=bool(row_id)), gr.update(visible=bool(row_id))

            def delete_row(row_id, df, prop_id):
                if not row_id or df is None or df.empty:
                    return df, gr.update(), gr.update(), gr.update(interactive=False, visible=False), gr.update(visible=False), df
                try:
                    selected_id = int(str(row_id))
                    match = df[df["ID"] == selected_id]
                except Exception:
                    match = pd.DataFrame()
                if match.empty:
                    return df, gr.update(), gr.update(), gr.update(interactive=False, visible=False), gr.update(visible=False), df

                row_id_val = int(match.iloc[0]["ID"])
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute('DELETE FROM accesspoint WHERE "ID" = %s', (row_id_val,))
                conn.commit()
                conn.close()
                df_updated, df_display = sync_accesspoints(prop_id)
                options = get_row_options(df_updated)
                return df_display, gr.update(choices=options, value=None), gr.update(choices=options, value=None), gr.update(interactive=False, visible=False), gr.update(visible=False), df_updated

            # ---- small UI helpers for chaining ----
            def _activate_selects_if_options(df):
                has_rows = (df is not None) and (not getattr(df, "empty", True))
                return (
                    gr.update(interactive=has_rows),
                    gr.update(interactive=has_rows),
                )

            def _sync_delete_interactive_on_edit(selected_val):
                return gr.update(interactive=not bool(selected_val))

            def _enable_selectors():
                return gr.update(interactive=True), gr.update(interactive=True)

            def _enable_selectors_and_add():
                return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

            def _disable_both_selectors():
                return gr.update(interactive=False), gr.update(interactive=False)

            def _toggle_edit_and_add_on_delete(row_choice):
                active = bool(row_choice)
                if active:
                    return gr.update(interactive=False), gr.update(interactive=False)
                else:
                    return gr.update(), gr.update()

            def _refresh_table_and_both_selects(prop_id):
                if not prop_id:
                    return gr.update(), None, gr.update(), gr.update()
                df_full, df_display = sync_accesspoints(prop_id)
                options = get_row_options(df_full)
                has_rows = (df_full is not None) and (not getattr(df_full, "empty", True))
                return (
                    gr.update(value=df_display),
                    df_full,
                    gr.update(choices=options, value=None, interactive=has_rows),
                    gr.update(choices=options, value=None, interactive=has_rows),
                )

            # ===== NEW: AP Groups helpers =====

            def _parse_apgroups_text(raw):
                """
                Parse text from property.APGroups into a list of (group_name, aps_string) tuples.
                Expects something like: [('Group 1', 'Front (123),'), ...]
                """
                groups: list[tuple[str, str]] = []
                if not raw:
                    return groups
                try:
                    val = ast.literal_eval(str(raw))
                    if isinstance(val, (list, tuple)):
                        for item in val:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                groups.append((str(item[0]), str(item[1])))
                except Exception as e:
                    print(f"⚠️ Failed to parse APGroups: {e}")
                return groups

            def _serialize_apgroups(groups):
                """
                Serialize list of (group_name, aps_string) tuples into a Python-literal string
                suitable for storing in property.APGroups.
                """
                if not groups:
                    return None
                try:
                    return repr([(str(name), str(aps)) for (name, aps) in groups])
                except Exception:
                    return repr(groups)

            def _load_apgroups_for_property(label):
                """
                When a property is selected, load its APGroups (if any) into state,
                and toggle Manage AP Groups button appropriately.
                """
                if not label or label not in prop_map:
                    # No valid property → keep groups empty and hide/disable Manage AP Groups
                    return [], gr.update(visible=False, interactive=False)
                prop_id, _pname, _owner = prop_map[label]
                if not prop_id:
                    return [], gr.update(visible=False, interactive=False)

                groups = []
                try:
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute(
                        'SELECT "APGroups" FROM property WHERE "PropID" = %s AND "UserID" = %s',
                        (int(prop_id), _uid())
                    )
                    row = cur.fetchone()
                    raw = row[0] if row and row[0] is not None else None
                    groups = _parse_apgroups_text(raw)
                except Exception as e:
                    print(f"⚠️ Failed to load APGroups for PropID={prop_id}: {e}")
                finally:
                    try:
                        cur.close(); conn.close()
                    except Exception:
                        pass

                # Enable AND show Manage AP Groups button when a property is selected
                return groups, gr.update(visible=True, interactive=True)

            def _build_apgroup_table_and_choices(groups, df_full):
                """
                Build DataFrame for AP Groups and choices for AP dropdown and delete dropdown.
                """
                if groups:
                    table = pd.DataFrame(groups, columns=["Group Name", "Access Points"])
                    delete_choices = [g[0] for g in groups]
                else:
                    table = pd.DataFrame(columns=["Group Name", "Access Points"])
                    delete_choices = []

                ap_choices: list[str] = []
                if df_full is not None and not getattr(df_full, "empty", True):
                    for _, row in df_full.iterrows():
                        nm = str(row.get("NameOfAccessPoint") or "").strip()
                        ino = str(row.get("ApInOut") or "").strip()     # add direction
                        apid = str(row.get("AccessPointID") or "").strip()
                        if not nm or not apid:
                            continue
                        label = f"{nm} {ino} ({apid})"
                        ap_choices.append(label)

                return (
                    table,
                    gr.update(choices=ap_choices, value=[]),
                    gr.update(choices=delete_choices, value=None),
                )

            def open_manage_apgroups(groups, df_full):
                """
                Open the Manage AP Groups panel, show current groups,
                and disable Add/Edit/Delete while managing.
                """
                table, aps_dd, del_dd = _build_apgroup_table_and_choices(groups, df_full)
                return (
                    gr.update(visible=True),      # apgroup_group visible
                    gr.update(value=table),       # apgroups_table
                    gr.update(value=""),          # apgroup_name
                    aps_dd,                       # apgroup_aps
                    del_dd,                       # apgroup_delete_dropdown
                    gr.update(interactive=False), # add_btn
                    gr.update(interactive=False), # manage_btn
                    gr.update(interactive=False), # row_to_edit
                    gr.update(interactive=False), # row_to_delete
                    gr.update(interactive=False), # delete_btn
                    gr.update(interactive=False), # apply_edit_btn
                )

            def save_apgroup(name, aps_selected, groups, df_full):
                """
                Create or update an AP Group (tuple row).
                name: group name
                aps_selected: list of 'AccessPointName (AccessPointID)' labels
                groups: current list of (name, aps_string)
                """
                name = (name or "").strip()
                if not name:
                    # No name → do nothing but refresh table
                    table, aps_dd, del_dd = _build_apgroup_table_and_choices(groups, df_full)
                    return table, groups, gr.update(value=name), aps_dd, del_dd

                labels = aps_selected or []
                # Build comma-separated string with trailing comma, e.g. "Front Gate (id1),Back Gate (id2),"
                aps_str = ""
                if labels:
                    aps_str = ",".join(labels) + ","

                new_groups: list[tuple[str, str]] = []
                replaced = False
                for gname, gval in groups:
                    if gname == name:
                        new_groups.append((name, aps_str))
                        replaced = True
                    else:
                        new_groups.append((gname, gval))
                if not replaced:
                    new_groups.append((name, aps_str))

                table, aps_dd, del_dd = _build_apgroup_table_and_choices(new_groups, df_full)
                return table, new_groups, gr.update(value=""), aps_dd, del_dd

            def _apgroup_delete_guard(del_name, groups, df_full, prop_id):
                """
                First stage when clicking 'Apply Delete of AP Group':
                  - If no group selected: just refresh table, hide warning/buttons, keep main buttons active.
                  - If group has NO usage: delete immediately and keep main buttons active.
                  - If group IS in use: show warning + confirm/cancel buttons, and disable
                    Save / Delete / Exit buttons until the user decides.
                """
                del_name = (del_name or "").strip()

                # Build table + delete choices from current groups
                if groups:
                    table = pd.DataFrame(groups, columns=["Group Name", "Access Points"])
                    delete_choices = [g[0] for g in groups]
                else:
                    table = pd.DataFrame(columns=["Group Name", "Access Points"])
                    delete_choices = []

                # Case 1: nothing selected → no-op, hide warning/buttons, keep main buttons active
                if not del_name:
                    return (
                        table,
                        groups,
                        gr.update(choices=delete_choices, value=None),   # delete dropdown
                        gr.update(value="", visible=False),              # warning
                        gr.update(visible=False),                        # cancel btn
                        gr.update(visible=False),                        # confirm btn
                        gr.update(interactive=True),                     # save btn
                        gr.update(interactive=True),                     # delete btn
                        gr.update(interactive=True),                     # exit btn
                    )

                # Check usage in parkingbays / tenantuser
                in_use, count_pb, count_tu = _check_apgroup_usage(del_name, prop_id)
                print(f"[DEBUG] APGroup delete guard: group='{del_name}', in_use={in_use}, pb={count_pb}, tu={count_tu}")

                # Case 2: safe -> delete immediately and keep main buttons active
                if not in_use:
                    new_groups = [g for g in groups if g[0] != del_name]
                    if new_groups:
                        new_table = pd.DataFrame(new_groups, columns=["Group Name", "Access Points"])
                        new_delete_choices = [g[0] for g in new_groups]
                    else:
                        new_table = pd.DataFrame(columns=["Group Name", "Access Points"])
                        new_delete_choices = []
                    return (
                        new_table,
                        new_groups,
                        gr.update(choices=new_delete_choices, value=None),
                        gr.update(value="", visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                    )

                # Case 3: in use -> show warning + confirm/cancel and disable main buttons
                warning_text = (
                    "NOTE: the AP Group you want to deleted is allocated to a Bay or Permission User. "
                    "If the AP Group is deleted then the AP Group will ALSO be removed from any Bay and/or "
                    "Permission User to which it has been allocated"
                )
                return (
                    table,
                    groups,
                    gr.update(choices=delete_choices, value=del_name),
                    gr.update(value=warning_text, visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(interactive=False),   # save disabled
                    gr.update(interactive=False),   # delete disabled
                    gr.update(interactive=False),   # exit disabled
                )

            def _apgroup_cancel_pending_delete(groups, df_full):
                """
                User clicked 'Cancel Delete' in the AP Group warning.
                Just refresh table & dropdowns and hide warning/buttons, re-enable main buttons.
                """
                if groups:
                    table = pd.DataFrame(groups, columns=["Group Name", "Access Points"])
                    delete_choices = [g[0] for g in groups]
                else:
                    table = pd.DataFrame(columns=["Group Name", "Access Points"])
                    delete_choices = []

                return (
                    table,
                    groups,
                    gr.update(choices=delete_choices, value=None),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=True),   # save enabled
                    gr.update(interactive=True),   # delete enabled
                    gr.update(interactive=True),   # exit enabled
                )

            def _apgroup_confirm_delete(del_name, groups, df_full, prop_id):
                """
                User clicked 'Proceed with Delete' after warning:
                  1) Delete AP group from list.
                  2) Remove group name from BayAccessNo (parkingbays).
                  3) Remove group name from PedestrianAPs (tenantuser).
                  4) Hide warning + confirm/cancel buttons.
                  5) Re-enable main buttons.
                """
                del_name = (del_name or "").strip()
                if not del_name:
                    # treat as cancel
                    return _apgroup_cancel_pending_delete(groups, df_full)

                # Cleanup all references first
                _cleanup_apgroup_references(del_name, prop_id)

                # Delete from groups list
                new_groups = [g for g in groups if g[0] != del_name]
                if new_groups:
                    table = pd.DataFrame(new_groups, columns=["Group Name", "Access Points"])
                    delete_choices = [g[0] for g in new_groups]
                else:
                    table = pd.DataFrame(columns=["Group Name", "Access Points"])
                    delete_choices = []

                return (
                    table,
                    new_groups,
                    gr.update(choices=delete_choices, value=None),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=True),   # save enabled
                    gr.update(interactive=True),   # delete enabled
                    gr.update(interactive=True),   # exit enabled
                )


            def exit_manage_apgroups(groups, prop_id):
                """
                Exit Manage AP Groups, saving the tuple list into property.APGroups for this PropID.
                """
                text_val = _serialize_apgroups(groups)
                if prop_id:
                    try:
                        conn = get_connection()
                        cur = conn.cursor()
                        cur.execute(
                            'UPDATE property SET "APGroups"=%s WHERE "PropID"=%s AND "UserID"=%s',
                            (text_val, int(prop_id), _uid())
                        )
                        conn.commit()
                    except Exception as e:
                        print(f"⚠️ Failed to save APGroups for PropID={prop_id}: {e}")
                    finally:
                        try:
                            cur.close(); conn.close()
                        except Exception:
                            pass

                # Hide panel and re-enable main controls
                return (
                    gr.update(visible=False),                # apgroup_group
                    gr.update(interactive=True),             # add_btn
                    gr.update(interactive=True),             # manage_btn
                    gr.update(interactive=True),             # row_to_edit
                    gr.update(interactive=True),             # row_to_delete
                    gr.update(interactive=False, visible=False),  # delete_btn
                    gr.update(visible=False),                # apply_edit_btn
                )

            def _toggle_manage_on_edit(row_choice):
                """
                Disable Manage AP Groups while an edit row is active.
                """
                return gr.update(interactive=not bool(row_choice))

            def _toggle_manage_on_delete(row_choice):
                """
                Disable Manage AP Groups while a delete row is active.
                """
                return gr.update(interactive=not bool(row_choice))

            def _disable_manage():
                return gr.update(interactive=False)

            def _enable_manage():
                return gr.update(interactive=True)

            # ---- UI wiring ----
            prop_map: Dict[str, tuple[int | None, str, str]] = {}
            print("DEBUG (prime): starting with UID =", _uid())

            # Start empty; demo.load populates it
            prop_dropdown = gr.Dropdown(label="Select Property (Owner)", choices=[], value=None)
            
            btn_reload_props = gr.Button("Reload Properties", variant="secondary")
            btn_reload_props.click(_prime_uid_then_fill_properties, outputs=[prop_dropdown])
            
            accesspoint_table = gr.Dataframe(label="Access Points at Property", visible=False, interactive=False)

            state_df = gr.State()
            state_prop_id = gr.State()
            state_apgroups = gr.State([])  # NEW: holds list of (GroupName, APsString) tuples

            demo.load(_prime_uid_then_fill_properties, inputs=[], outputs=[prop_dropdown])

            with gr.Row():
                add_btn = gr.Button("Add New Access Point", interactive=False)
                # CHANGED: start Manage AP Groups button hidden until a property is selected
                manage_btn = gr.Button("Manage AP Groups", interactive=False, visible=False)  # NEW behaviour

            with gr.Row():
                row_to_edit = gr.Dropdown(label="Select Access Point to Edit", choices=[], value=None, interactive=False)
                row_to_delete = gr.Dropdown(label="Select Access Point to Delete", choices=[], value=None, interactive=False)

            with gr.Row():
                delete_btn = gr.Button("Apply Delete", interactive=False, visible=False)
                cancel_delete_btn = gr.Button("Cancel Delete", visible=False)

            with gr.Group(visible=False) as edit_group:
                gr.Markdown("""<div style="background-color:#666666; color:#FFFFFF; padding:10px; border-radius:0px;">Access Point Details</div>""")
                with gr.Row():
                    edit_phone = gr.Textbox(label="Access Point ID (auto)", interactive=False)
                    edit_name = gr.Textbox(label="Name of Access Point")
                    edit_inout = gr.Dropdown(label="Direction (In | Out | N/A)", choices=["In","Out", "N/A"], value=None)
                    
                # NEW: APOpen / APClose editors + APActive checkbox (next to Close)
                with gr.Row():
                    edit_restrict = gr.Checkbox(label="Tick if Restricted AP")
                    edit_ap_open = gr.Textbox(label="AP Opens at (HH:MM)", placeholder="HH:MM")
                    edit_ap_close = gr.Textbox(label="AP Closes at (HH:MM)", placeholder="HH:MM")
                    edit_ap_active = gr.Checkbox(label="AP Active?")  # <— NEW checkbox

                gr.Markdown("""<div style="background-color:#666666; color:#FFFFFF; padding:10px; border-radius:0px;">Access Point Installer details</div>""")
                with gr.Row():
                    edit_installer_dd = gr.Dropdown(
                        label="Installer (select company)",
                        choices=load_installer_options(),
                        value=None
                    )
                    edit_installer_id = gr.Textbox(label="Installer ID (auto)", interactive=False, visible=False)
                    edit_device = gr.Textbox(label="Installer's Device Name (auto))", interactive=False)
                with gr.Row():
                    error_msg = gr.Markdown(visible=False)

            with gr.Row():
                apply_edit_btn = gr.Button("Apply Edit", interactive=False, visible=False)
                cancel_edit_btn = gr.Button("Cancel Edit", visible=False)
                download_qr_btn = gr.DownloadButton(
                   label="Download QR",
                   visible=False,  # start hidden
               )

            # NEW: Manage AP Groups UI
            with gr.Group(visible=False) as apgroup_group:
                gr.Markdown(
                    """<div style="background-color:#666666; color:#FFFFFF; padding:10px; border-radius:0px;">
                    Management of Access Point Groups for this Property
                    </div>"""
                )
                apgroups_table = gr.Dataframe(
                    headers=["Add new Group Name", "Add Access Points to new Group"],
                    value=pd.DataFrame(columns=["Group Name", "Access Points"]),
                    interactive=False
#                    label="Manage Access Point Groups at Property"
                )
                with gr.Row():
                    apgroup_name = gr.Textbox(label="New Group Name")
                    apgroup_aps = gr.Dropdown(
                        label="Add Access Points to New Group",
                        choices=[],
                        multiselect=True,
                        value=[]
                    )
                    apgroup_delete_dropdown = gr.Dropdown(
                        label="Select AP Group to Delete",
                        choices=[],
                        value=None
                    )
                with gr.Row():
                    apgroup_save_btn = gr.Button("Save New Group")
                    apgroup_exit_btn = gr.Button("Done / Exit")
                    apgroup_delete_btn = gr.Button("Apply Delete of AP Group")

                # Warning + confirm/cancel for AP Group delete
                apgroup_warning = gr.Markdown(value="", visible=False, elem_id="apgroup-warning-box")
#                apgroup_warning = gr.Markdown(value="", visible=False)
                with gr.Row():
                    apgroup_cancel_delete_btn = gr.Button("Cancel Delete", visible=False)
                    apgroup_confirm_delete_btn = gr.Button("Proceed with Delete", visible=False)


            # events
            prop_dropdown.change(
                select_property,
                inputs=prop_dropdown,
                outputs=[
                    accesspoint_table, state_df, state_prop_id,
                    row_to_edit, row_to_delete,
                    apply_edit_btn, delete_btn, add_btn,
                    edit_group, error_msg, cancel_edit_btn, cancel_delete_btn
                ],
            ).then(
                _activate_selects_if_options,
                inputs=[state_df],
                outputs=[row_to_edit, row_to_delete],
            ).then(
                _load_apgroups_for_property,    # NEW: load APGroups for this property
                inputs=[prop_dropdown],
                outputs=[state_apgroups, manage_btn],
            )

            edit_installer_dd.change(installer_changed, inputs=edit_installer_dd, outputs=edit_installer_id)

            row_to_edit.change(
                select_row,
                inputs=[row_to_edit, state_df],
                outputs=[
                    edit_name, edit_phone, edit_restrict, edit_device, edit_inout,
                    edit_installer_dd, edit_installer_id,
                    edit_group, apply_edit_btn, add_btn, delete_btn, cancel_edit_btn,
                    download_qr_btn, edit_ap_open, edit_ap_close, edit_ap_active,
                ]
            ).then(
                _sync_delete_interactive_on_edit,
                inputs=[row_to_edit],
                outputs=[row_to_delete]
            ).then(
                _toggle_manage_on_edit,      # NEW: disable Manage while editing
                inputs=[row_to_edit],
                outputs=[manage_btn]
            )

            cancel_edit_btn.click(
                cancel_edit,
                outputs=[
                    edit_name, edit_phone, edit_restrict, edit_device, edit_inout,
                    edit_installer_dd, edit_installer_id, edit_group, apply_edit_btn,
                    add_btn, delete_btn, cancel_edit_btn, edit_ap_open, edit_ap_close, edit_ap_active
                ]
            ).then(
                _refresh_table_and_both_selects,
                inputs=[state_prop_id],
                outputs=[accesspoint_table, state_df, row_to_edit, row_to_delete]
            ).then(
                _enable_selectors, inputs=[], outputs=[row_to_edit, row_to_delete]
            ).then(
                _enable_manage, inputs=[], outputs=[manage_btn]   # NEW: re-enable Manage
            )

            row_to_delete.change(
                enable_delete_button, inputs=row_to_delete, outputs=[delete_btn, cancel_delete_btn]
            ).then(
                _toggle_edit_and_add_on_delete, inputs=[row_to_delete], outputs=[row_to_edit, add_btn]
            ).then(
                _toggle_manage_on_delete, inputs=[row_to_delete], outputs=[manage_btn]  # NEW: disable Manage
            )

            cancel_delete_btn.click(
                cancel_delete,
                outputs=[row_to_delete, delete_btn, add_btn, cancel_delete_btn]
            ).then(
                _enable_selectors, inputs=[], outputs=[row_to_edit, row_to_delete]
            ).then(
                _enable_manage, inputs=[], outputs=[manage_btn]   # NEW: re-enable Manage
            )

            apply_edit_btn.click(
                apply_edit,
                inputs=[
                    row_to_edit, state_df, edit_name, edit_phone, edit_restrict, edit_device,
                    edit_inout, edit_installer_dd, edit_installer_id, state_prop_id,
                    edit_ap_open, edit_ap_close, edit_ap_active,   # NEW input
                ],
                outputs=[
                    accesspoint_table,  # updated table
                    row_to_edit,
                    row_to_delete,
                    edit_name, edit_phone, edit_restrict, edit_device, edit_inout,
                    edit_installer_dd, edit_installer_id,
                    edit_group, error_msg,
                    apply_edit_btn, add_btn, delete_btn, cancel_edit_btn,
                    download_qr_btn,
                    edit_ap_open, edit_ap_close, edit_ap_active,   # NEW outputs (3)
                ]
            ).then(
                _refresh_table_and_both_selects,
                inputs=[state_prop_id],
                outputs=[accesspoint_table, state_df, row_to_edit, row_to_delete]
            ).then(
                _enable_selectors, inputs=[], outputs=[row_to_edit, row_to_delete]
            ).then(
                _enable_manage, inputs=[], outputs=[manage_btn]   # NEW: re-enable Manage
            )

            add_btn.click(
                _disable_both_selectors, inputs=[], outputs=[row_to_edit, row_to_delete]
            ).then(
                _disable_manage, inputs=[], outputs=[manage_btn]   # NEW: disable Manage during Add
            ).then(
                add_row,
                inputs=state_prop_id,
                outputs=[accesspoint_table, row_to_edit, row_to_delete, row_to_edit, state_df]
            ).then(
                _enable_selectors, inputs=[], outputs=[row_to_edit, row_to_delete]
            ).then(
                _enable_manage, inputs=[], outputs=[manage_btn]    # NEW: re-enable Manage
            )

            delete_btn.click(
                delete_row, inputs=[row_to_delete, state_df, state_prop_id],
                outputs=[accesspoint_table, row_to_edit, row_to_delete, delete_btn, cancel_delete_btn, state_df]
            ).then(
                _enable_selectors_and_add, inputs=[], outputs=[row_to_edit, row_to_delete, add_btn]
            ).then(
                _enable_manage, inputs=[], outputs=[manage_btn]    # NEW: re-enable Manage
            )

            # NEW: Manage AP Groups wiring
            manage_btn.click(
                open_manage_apgroups,
                inputs=[state_apgroups, state_df],
                outputs=[
                    apgroup_group,
                    apgroups_table,
                    apgroup_name,
                    apgroup_aps,
                    apgroup_delete_dropdown,
                    add_btn,
                    manage_btn,
                    row_to_edit,
                    row_to_delete,
                    delete_btn,
                    apply_edit_btn,
                ]
            )

            apgroup_save_btn.click(
                save_apgroup,
                inputs=[apgroup_name, apgroup_aps, state_apgroups, state_df],
                outputs=[apgroups_table, state_apgroups, apgroup_name, apgroup_aps, apgroup_delete_dropdown],
            )

            apgroup_delete_btn.click(
                _apgroup_delete_guard,
                inputs=[apgroup_delete_dropdown, state_apgroups, state_df, state_prop_id],
                outputs=[
                    apgroups_table,             # updated table
                    state_apgroups,            # updated groups state
                    apgroup_delete_dropdown,   # dropdown (may keep or clear selection)
                    apgroup_warning,           # warning text
                    apgroup_cancel_delete_btn, # cancel button (visible/hidden)
                    apgroup_confirm_delete_btn,# confirm button (visible/hidden)
                    apgroup_save_btn,          # Save New Group button (enabled/disabled)
                    apgroup_delete_btn,        # Apply Delete of AP Group button (enabled/disabled)
                    apgroup_exit_btn,          # Done / Exit button (enabled/disabled)
                ],
            )

            apgroup_cancel_delete_btn.click(
                _apgroup_cancel_pending_delete,
                inputs=[state_apgroups, state_df],
                outputs=[
                    apgroups_table,             # table
                    state_apgroups,            # groups state (unchanged)
                    apgroup_delete_dropdown,   # dropdown (cleared)
                    apgroup_warning,           # warning hidden
                    apgroup_cancel_delete_btn, # cancel button hidden
                    apgroup_confirm_delete_btn,# confirm button hidden
                    apgroup_save_btn,          # Save New Group re-enabled
                    apgroup_delete_btn,        # Apply Delete re-enabled
                    apgroup_exit_btn,          # Done / Exit re-enabled
                ],
            )

            apgroup_confirm_delete_btn.click(
                _apgroup_confirm_delete,
                inputs=[apgroup_delete_dropdown, state_apgroups, state_df, state_prop_id],
                outputs=[
                    apgroups_table,             # updated table after delete
                    state_apgroups,            # updated groups state
                    apgroup_delete_dropdown,   # dropdown (cleared)
                    apgroup_warning,           # warning hidden
                    apgroup_cancel_delete_btn, # cancel button hidden
                    apgroup_confirm_delete_btn,# confirm button hidden
                    apgroup_save_btn,          # Save New Group re-enabled
                    apgroup_delete_btn,        # Apply Delete re-enabled
                    apgroup_exit_btn,          # Done / Exit re-enabled
                ],
            )


            apgroup_exit_btn.click(
                exit_manage_apgroups,
                inputs=[state_apgroups, state_prop_id],
                outputs=[apgroup_group, add_btn, manage_btn, row_to_edit, row_to_delete, delete_btn, apply_edit_btn],
            )

    return demo

# app = launch_accesspoints_module()
# app.launch(share=True)
