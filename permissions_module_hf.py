import gradio as gr
import psycopg2
import pandas as pd
import random
import re
from pathlib import Path
import pycountry
from phonenumbers import COUNTRY_CODE_TO_REGION_CODE
import ast
import pickle
from cryptography.fernet import Fernet, InvalidToken
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet, InvalidToken  # not just Fernet
import time
import json
import os

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

    # Prefer JSON; allow JSON; allow Python-literal fallback
    try:
        params = json.loads(decrypted)
    except json.JSONDecodeError:
        params = ast.literal_eval(decrypted)  # handles "{'host': ...}" format

    return params


def get_connection():
    params = decrypt_nhost()
    return psycopg2.connect(params)


# ─────────────────────────────────────────────
# Active User: same approach as property_module_hf.py
# ─────────────────────────────────────────────

# encrypted-pickle fallbacks (same names used in your other modules)
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




def launch_permissions_module(get_user_id=None):

    if _coerce_uid(ActiveUserID):
        return ActiveUserID

    
    # ========== UI ==========
    with gr.Blocks() as demo:

        with gr.Tab("Manage Users that have Access Point Permissions"):

            # ----- Debug: UID + DB connectivity -----

            def _uid() -> Optional[int]:
                # convenience wrapper used throughout the module
                return resolve_and_cache_uid(get_user_id)
            
            def get_country_intcode():
                country_to_code = {}
                for code, regions in COUNTRY_CODE_TO_REGION_CODE.items():
                    for region in regions:
                        country = pycountry.countries.get(alpha_2=region)
                        if country:  # skip unrecognized codes
                            country_to_code[country.name] = f"+{code}"
            
                # Sort by country name
                country_to_code = dict(sorted(country_to_code.items()))
            
                # Force SA at the top
                sorted_country_to_code = {
                    "South Africa": country_to_code.get("South Africa", "+27"),
                    **{k: v for k, v in country_to_code.items() if k != "South Africa"}
                }
                return sorted_country_to_code

            # Country dropdown data & helpers
            COUNTRY_MAP = get_country_intcode()  # {'South Africa': '+27', 'Albania': '+355', ...}
            COUNTRY_CHOICES = [f"{name} ({code})" for name, code in COUNTRY_MAP.items()]
            DEFAULT_COUNTRY_LABEL = next((lbl for lbl in COUNTRY_CHOICES if lbl.startswith("South Africa")), COUNTRY_CHOICES[0])
            
            def _country_label_from_db(country_val: Any) -> str:
                """Map DB stored smallint-like (e.g. 27) to dropdown label 'South Africa (+27)'."""
                digits = str(country_val or "").lstrip("+").strip()
                if not digits:
                    return DEFAULT_COUNTRY_LABEL
                code = f"+{digits}"
                for name, c in COUNTRY_MAP.items():
                    if c == code:
                        return f"{name} ({c})"
                return DEFAULT_COUNTRY_LABEL
            
            def _extract_country_digits(label: str | None) -> str:
                """From 'South Africa (+27)' -> '27' (digits only)"""
                m = re.search(r"\+?(\d+)", label or "")
                return m.group(1) if m else ""
      
        
            # ========== UTILS ==========
            def generate_id():
                return random.randint(111111111111111, 999999999999999)
        
            def _lic_or_null(val):
                s = "" if val is None else str(val).strip()
                return None if s == "" or s == "()" else s
        
            def _mobile_or_null(val):
                s = "" if val is None else str(val).strip()
                return None if s == "" else s
        
            def is_valid_mobile(mobile):    #need to agree this with Luke
                if mobile is None:
                    return True
                mobile = str(mobile).strip()
                if len(mobile) == 9 and mobile.isdigit():
                    mobile = '0' + mobile
                return bool(re.fullmatch(r"(?:\+27|27|0)[6-8][0-9]{8}", mobile))
        
            def _mobile_to_db(mobile_str: str | None):
                s = "" if mobile_str is None else str(mobile_str).strip()
                if s == "":
                    return None
                num_like = s.lstrip("+-")
                if num_like.isdigit():
                    try:
                        if int(s) <= 0:
                            return None
                    except Exception:
                        pass
                return s


            # --- NEW: APGroups parser ---------------------------------------
            def _parse_apgroups_text(raw):
                """
                Parse property.APGroups text into a list of (group_name, aps_string) tuples.
                Accepts Python-literal or list/tuple-like structures.
                """
                groups = []
                if not raw:
                    return groups
                try:
                    val = ast.literal_eval(str(raw))
                    if isinstance(val, (list, tuple)):
                        for item in val:
                            if isinstance(item, (list, tuple)) and len(item) >= 1:
                                name = str(item[0])
                                aps_str = str(item[1]) if len(item) > 1 else ""
                                groups.append((name, aps_str))
                except Exception as e:
                    print(f"⚠️ Failed to parse APGroups: {e}")
                return groups

        
            def _mobile_display(val, *, null_as_zero=True):
                import math
                s = "" if val is None else str(val).strip()
                if s == "" or s.lower() == "nan":
                    return "0" if null_as_zero else ""
                try:
                    f = float(s)
                    if math.isfinite(f) and f.is_integer():
                        return str(int(f))
                except Exception:
                    pass
                if s.endswith(".0") and s[:-2].isdigit():
                    return s[:-2]
                return s
        
            def fetch_property_options():
                uid = _uid()
                print(f"DEBUG load_property(): ActiveUserID={uid}")
                if uid is None:
                    return pd.DataFrame(columns=["PropID", "Name", "Owner"])
            
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT "PropID", "Name", "Owner" FROM property '
                    'WHERE "UserID" = %s AND "Active" = TRUE',
                    (uid,)
                )
                rows = cursor.fetchall()
                conn.close()
            
                # Return a DataFrame so .iterrows() works below
                return pd.DataFrame(rows, columns=["PropID", "Name", "Owner"])
        
            def sync_tenantuser(prop_id):
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT "ID", "PropID", "TenantUserName", "TenantMobileNo", "Country", "UserID", "LeaseID", '
                    '        "BayAllocated", "PedestrianAPs", "CarLicenceA", "CarLicenceB", "Active", "AfterHours", "AllAPs" '
                    'FROM tenantuser WHERE "UserID" = %s AND "PropID" = %s',
                    (_uid(), int(prop_id))
                )
                rows = cursor.fetchall()
                conn.close()
            
                columns = [
                    "ID", "PropID", "TenantUserName", "TenantMobileNo", "Country", "UserID", "LeaseID",
                    "BayAllocated", "PedestrianAPs", "CarLicenceA", "CarLicenceB", "Active", "AfterHours", "AllAPs"
                ]
                df = pd.DataFrame(rows, columns=columns)
            
                display_columns = [                    
                    "TenantUserName", 
                    "Active", 
                    "AfterHours",
                    "AllAPs",
                    "Country", 
                    "TenantMobileNo",
                    "LeaseID", 
                    "BayAllocated", 
                    "PedestrianAPs",

                    "CarLicenceA", 
                    "CarLicenceB",
                ]
                display_renames = {
                    "TenantUserName": "User's Name",
                    "Active": "Active",
                    "AfterHours": "A/H Access",
                    "AllAPs": "All APs",
                    "Country": "Code",
                    "TenantMobileNo": "Mobile No",
                    "LeaseID": "Lease / Permission Type",
                    "BayAllocated": "Bay",
                    "PedestrianAPs": "AP Groups",
                    "CarLicenceA": "Car Licence A",
                    "CarLicenceB": "Car Licence B",
                }
            
                if df.empty:
                    df_display = pd.DataFrame(columns=[display_renames[c] for c in display_columns])
                else:
                    df = df.sort_values(by="LeaseID", ascending=True, key=lambda x: x.astype(str))
                    df_display = df[display_columns].rename(columns=display_renames)
            
                return df, df_display


            def get_available_bays(prop_id, lease_id):
                print("DEBUG - Lease ID", lease_id)
                if isinstance(lease_id, str) and "(" in lease_id and ")" in lease_id:
                    lease_id = lease_id.split("(")[-1].replace(")", "").strip()
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT "BayNo", "ID" FROM parkingbays WHERE "PropID" = %s AND "LeaseID" = %s',
                    (prop_id, lease_id)
                )
                all_bays = cursor.fetchall()
                cursor.execute(
                    'SELECT "BayAllocated" FROM tenantuser '
                    'WHERE "PropID" = %s AND POSITION(LOWER(%s) IN LOWER("LeaseID")) > 0 AND "Active" = TRUE',
                    (prop_id, lease_id)
                )
                used_bays = {b[0].split(" (")[0] for b in cursor.fetchall() if b[0]}
                conn.close()
                return [f"{bay_no}" for bay_no, _ in all_bays if bay_no not in used_bays]
        
            def get_restricted_access_points(prop_id):
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT "NameOfAccessPoint", "ID" FROM accesspoint '
                    'WHERE "PropID" = %s AND "RestrictedAP" = TRUE',
                    (prop_id,)
                )
                rows = cursor.fetchall()
                conn.close()
                return [f"{name} ({id})" for name, id in rows] if rows else []
        
            def get_unique_lease_options(prop_id):
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT DISTINCT "TenantName", "LeaseID" FROM parkingbays WHERE "PropID" = %s',
                    (int(prop_id),)
                )
                rows = cursor.fetchall()
                conn.close()
            
                unique_set = set()
                for tenant_name, lease_id in rows:
                    tn = (tenant_name or "").strip()
                    li = (lease_id or "").strip()
                    if tn and li:
                        unique_set.add(f"{tn} ({li})")
            
                base = sorted(s for s in unique_set)
                return base
        
                    
                    
            def _normalize_tenantuser_columns(df: pd.DataFrame) -> pd.DataFrame:

                if df is None or df.empty:
                    expected = ["LeaseID", "TenantUserName", "TenantMobileNo", "CarLicenceA", "CarLicenceB"]
                    for col in expected:
                        if col not in df.columns:
                            df[col] = []
                    return df
                alias_map = {
                    "TenantName": "TenantUserName",
                    "TenantUserMobile": "TenantMobileNo",
                    "MobileNo": "TenantMobileNo",
                }
                for old, new in alias_map.items():
                    if old in df.columns and new not in df.columns:
                        df[new] = df[old]
                required = ["LeaseID", "TenantUserName", "TenantMobileNo", "CarLicenceA", "CarLicenceB"]
                for col in required:
                    if col not in df.columns:
                        df[col] = ""
                return df
        
            def safe_refresh_bay_selection(selected_bay, lease_selected, edit_lease_val, prop_id):
                lease_input = lease_selected or edit_lease_val
                if lease_input and prop_id:
                    lease_id = lease_input.split("(")[-1].replace(")", "").strip()
                    bay_list = get_available_bays(prop_id, lease_id)
                    if not bay_list:
                        return (gr.update(visible=True), gr.update(value=""))
                    updated_value = selected_bay if selected_bay in bay_list else None
                    return (gr.update(choices=bay_list, value=updated_value, visible=True),
                            gr.update(value=updated_value or ""))
                return (gr.update(choices=[], value=None, visible=True), gr.update(value=""))


            def uid_conn_check():
                uid = _uid()
                # Try a trivial DB ping
                try:
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute("SELECT 1")
                    ok = cur.fetchone()
                    conn_txt = f"OK (SELECT 1 → {ok[0] if ok else 'None'})"
                except Exception as e:
                    conn_txt = f"ERROR: {e}"
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
                return f"**ActiveUserID**: `{_uid()}`  ·  **DB**: {conn_txt}"

            def uid_sources_check():
                env_uid = os.getenv("OPENQR_ACTIVE_USER_ID") or os.getenv("OQR_ACTIVE_USER_ID")
                cb_uid = None
                try:
                    if callable(get_user_id):
                        cb_uid = get_user_id()
                except Exception as e:
                    cb_uid = f"error: {e}"
                return "Env UID: `{}` · Callback UID: `{}` · ActiveUserID: `{}`".format(
                    env_uid, cb_uid, _uid()
                )

            apgroups_by_prop = {}  # cache APGroups per PropID

            def _prime_uid_then_fill_properties():

                deadline = time.time() + 10.0
                while resolve_and_cache_uid(get_user_id) is None and time.time() < deadline:
                    time.sleep(0.1)
            
                uid = _uid()
                print("DEBUG (prime): starting with UID =", uid)
            
                # If still no UID, keep the dropdown empty & disabled
                if uid is None:
                    return gr.update(choices=[], value=None, interactive=False)
            
                # Now UID is present → fetch and populate
                df = fetch_property_options()
                options = [f"{row['Name']} ({row['Owner']})" for _, row in df.iterrows()]
            
                # refresh the shared map in-place: label -> (PropID, Name)
                new_map = {opt: (int(row["PropID"]), row["Name"]) for opt, (_, row) in zip(options, df.iterrows())}
                prop_map.clear()
                prop_map.update(new_map)

                # Reset APGroups cache on full property reload (mutate; do NOT rebind)
                apgroups_by_prop.clear()
                
                # Enable dropdown if we have options
                return gr.update(choices=options, value=None, interactive=bool(options))
                        

# ---------- HANDLERS ----------

            def get_row_options(df):
                if df is None or df.empty:
                    return []
                def is_active(v):
                    if isinstance(v, bool):
                        return v
                    s = str(v).strip().lower()
                    return s in ("true", "t", "1", "yes", "y")
                all_df = df.copy()
                all_df["__inactive"] = ~all_df["Active"].apply(is_active)
                all_df["LeaseID_str"] = all_df["LeaseID"].astype(str)
                all_df = all_df.sort_values(by="LeaseID_str", ascending=True)
                return [
                    f"{row['LeaseID']} ({row['TenantUserName']})" + (" [inactive]" if row["__inactive"] else "")
                    for _, row in all_df.iterrows()
                ]

            def _display_key_series(df):
                def is_active(v):
                    if isinstance(v, bool):
                        return v
                    s = str(v).strip().lower()
                    return s in ("true", "t", "1", "yes", "y")
                return (
                    df["LeaseID"].astype(str) + " (" + df["TenantUserName"].astype(str) + ")" +
                    df["Active"].apply(lambda x: "" if is_active(x) else " [inactive]")
                )

            # NEW helper: distinct LeaseID options for "Remove Lease Permissions"
            def _get_leaseid_remove_options_for_prop(prop_id: Optional[int]) -> list[str]:
                if not prop_id:
                    return []
                try:
                    conn = get_connection()
                    cur = conn.cursor()
                    cur.execute(
                        'SELECT DISTINCT "LeaseID" FROM tenantuser '
                        'WHERE "UserID" = %s AND "PropID" = %s AND "LeaseID" IS NOT NULL',
                        (_uid(), int(prop_id))
                    )
                    rows = cur.fetchall()
                    conn.close()
                    leases = sorted(
                        {str(r[0]).strip() for r in rows if r[0] not in (None, "", "nan", "NaN")}
                    )
                    return leases
                except Exception:
                    return []

            def select_property(value):

                # outputs: 29
                if not value or value not in prop_map:
                    return [
                        gr.update(visible=False),  # tenant_table
                        None,                      # state_df
                        None,                      # state_prop_id
                        gr.update(visible=False),  # row_to_edit
                        gr.update(choices=[], value=None,
                                  visible=False, interactive=False),        # remove_lease_dropdown
                        gr.update(visible=False),  # row_to_delete
                        gr.update(visible=False),  # edit_group
                        gr.update(value=""), gr.update(value=""), gr.update(value=""),  # edit_name, edit_mobile, edit_country
                        gr.update(value=""), gr.update(value=""),                       # car_a, car_b
                        gr.update(value="", interactive=False),                         # edit_lease
                        gr.update(value=""),                                            # edit_bay
                        gr.update(value=""),                                            # edit_aps
                        gr.update(value=False),                                         # edit_active
                        gr.update(value=False),                                         # edit_afterhours
                        gr.update(value=False),                                         # edit_allaps
                        gr.update(value="", visible=False),                             # hidden_id
                        gr.update(choices=[], visible=False),                           # bay_dropdown
                        gr.update(choices=[], visible=False),                           # aps_dropdown
                        gr.update(choices=[], visible=False),                           # lease_dropdown
                        gr.update(visible=False),                                       # apply_edit_btn
                        gr.update(visible=False, interactive=False),                    # add_btn
                        gr.update(visible=False),                                       # delete_btn
                        gr.update(value=DEFAULT_COUNTRY_LABEL,
                                  interactive=False, visible=False),                    # add_country
                        gr.update(value="", interactive=False, visible=False),          # add_mobile
                        gr.update(visible=False, interactive=False),                    # process_new_btn
                        gr.update(visible=False, interactive=False),                    # cancel_new_btn
                    ]

                # Lookup from prop_map
                prop_id, _prop_name = prop_map[value]
                apgroups = []
                try:
                    conn_ap = get_connection()
                    cur_ap = conn_ap.cursor()
                    cur_ap.execute(
                        'SELECT "APGroups" FROM property WHERE "PropID" = %s AND "UserID" = %s',
                        (int(prop_id), _uid())
                    )
                    row = cur_ap.fetchone()
                    raw = row[0] if row and row[0] is not None else None
                    apgroups = _parse_apgroups_text(raw)
                    conn_ap.close()
                except Exception as e:
                    print(f"⚠️ Failed to load APGroups for PropID={prop_id}: {e}")
                apgroups_by_prop[int(prop_id)] = apgroups

                df, df_display = sync_tenantuser(prop_id)
                options = get_row_options(df)  # ALL rows
                lease_remove_opts = _get_leaseid_remove_options_for_prop(prop_id)

                return [
                    gr.update(value=df_display, visible=True),                 # tenant_table
                    df, prop_id,                                               # state_df, state_prop_id
                    gr.update(choices=options, value=None,
                              visible=True, interactive=True),                 # row_to_edit ACTIVE
                    gr.update(
                        choices=lease_remove_opts,
                        value=None,
                        visible=True,
                        interactive=True
                    ),                                                         # remove_lease_dropdown
                    gr.update(choices=options, value=None,
                              visible=True, interactive=True),                 # row_to_delete ACTIVE
                    gr.update(visible=False),                                  # edit_group
                    gr.update(value=""), gr.update(value=""), gr.update(value=""),
                    gr.update(value=""), gr.update(value=""),
                    gr.update(value="", interactive=False),
                    gr.update(value=""), gr.update(value=""),
                    gr.update(value=False),              # edit_active
                    gr.update(value=False),              # edit_afterhours
                    gr.update(value=False),              # edit_allaps
                    gr.update(value="", visible=False),
                    gr.update(choices=[], visible=False),
                    gr.update(choices=[], visible=False),
                    gr.update(choices=[], visible=False),
                    gr.update(visible=False),                                  # apply_edit_btn
                    gr.update(visible=True, interactive=True),                 # add_btn ACTIVE
                    gr.update(visible=False),                                  # delete_btn
                    gr.update(value=DEFAULT_COUNTRY_LABEL,
                              interactive=False, visible=False),               # add_country (hidden)
                    gr.update(value="", interactive=False, visible=False),     # add_mobile (hidden)
                    gr.update(visible=False, interactive=False),               # process_new_btn
                    gr.update(visible=False, interactive=False),               # cancel_new_btn
                ]



            def select_row(row_id, df, prop_id):
                # outputs: 21
                def _clear_ui():
                    return (
                        [gr.update(value="")] * 8 +  # edit_name, edit_mobile, edit_country, car_a, car_b, lease, bay, aps
                        [
                            gr.update(value=False),             # edit_active
                            gr.update(value=False),             # edit_afterhours
                            gr.update(value=False),             # edit_allaps
                            gr.update(value="", visible=False), # hidden_id
                            gr.update(choices=[], visible=False),  # bay_dropdown
                            gr.update(choices=[], visible=False),  # aps_dropdown
                            gr.update(choices=[], visible=False),  # lease_dropdown
                            gr.update(visible=False),              # edit_group
                            gr.update(visible=False),              # apply_edit_btn
                            gr.update(visible=True, interactive=True),  # add_btn
                            gr.update(visible=False, interactive=False), # delete_btn
                            gr.update(visible=False),              # cancel_edit_btn
                            gr.update(visible=True, interactive=True),  # row_to_delete
                        ]
                    )


                if not row_id or df is None or df.empty:
                    return _clear_ui()

                def _attach_display(_df: pd.DataFrame) -> pd.DataFrame:
                    _df = _df.copy()
                    _df["Display"] = _display_key_series(_df)
                    return _df

                df_all = _attach_display(df)
                sel = df_all[df_all["Display"].astype(str).str.strip() == (row_id or "").strip()]

                if sel.empty:
                    df_live, _ = sync_tenantuser(prop_id)
                    if df_live is None or df_live.empty:
                        try: gr.Info("That row no longer exists. Please reselect.")
                        except Exception: pass
                        return _clear_ui()
                    df_live = _attach_display(df_live)
                    sel = df_live[df_live["Display"].astype(str).str.strip() == (row_id or "").strip()]
                    if sel.empty:
                        try: gr.Info("That row changed or was removed. Please reselect.")
                        except Exception: pass
                        return _clear_ui()

                row = sel.iloc[0]
                lease_id = row["LeaseID"]
                bay_allocated = row["BayAllocated"] or ""
                pedestrian_aps = row["PedestrianAPs"] or ""
                car_a_val = row["CarLicenceA"] or ""
                car_b_val = row["CarLicenceB"] or ""
                active_val = bool(row["Active"])
                afterhours_val = bool(row.get("AfterHours", False))
                allaps_val = bool(row.get("AllAPs", False))

                bay_list = get_available_bays(prop_id, lease_id)
                groups = apgroups_by_prop.get(int(prop_id), []) if prop_id else []
                if groups:
                    # Only the group name (first element of each tuple)
                    ap_list = [g[0] for g in groups]
                else:
                    # Fallback to individual restricted access points if no groups stored
                    ap_list = get_restricted_access_points(prop_id)
                lease_options = get_unique_lease_options(prop_id)

                lease_dropdown_val = None
                for item in lease_options:
                    if item.endswith(f"({lease_id})"):
                        lease_dropdown_val = item
                        break

                return [
                    gr.update(value=row["TenantUserName"], interactive=False),                # edit_name [auto]
                    gr.update(
                        value=_mobile_display(row["TenantMobileNo"], null_as_zero=True),
                        interactive=False
                    ),                                                                        # edit_mobile [auto]
                    gr.update(
                        value=_country_label_from_db(row.get("Country")),
                        interactive=False
                    ),                                                                        # edit_country [auto]
                    gr.update(value=car_a_val, interactive=True),                            # edit_car_a
                    gr.update(value=car_b_val, interactive=True),                            # edit_car_b
                    gr.update(value=row["LeaseID"], interactive=False),                      # edit_lease
                    gr.update(value=bay_allocated, interactive=False),                       # edit_bay
                    gr.update(value=pedestrian_aps, interactive=False),                      # edit_aps
                    gr.update(value=active_val),                                             # edit_active
                    gr.update(value=afterhours_val),                                         # edit_afterhours
                    gr.update(value=allaps_val),                                             # edit_allaps
                    gr.update(value=str(row["ID"]), visible=False),                          # hidden_id
                    gr.update(
                        choices=bay_list,
                        value=(bay_allocated if bay_allocated in bay_list else None),
                        visible=True
                    ),                                                                       # bay_dropdown
                    gr.update(
                        choices=ap_list,
                        value=(pedestrian_aps.split(", ") if pedestrian_aps else []),
                        visible=True
                    ),                                                                       # aps_dropdown
                    gr.update(
                        choices=lease_options,
                        value=(lease_dropdown_val if lease_dropdown_val in lease_options else None),
                        visible=True
                    ),                                                                       # lease_dropdown
                    gr.update(visible=True),                                                 # edit_group
                    gr.update(visible=True),                                                 # apply_edit_btn
                    gr.update(visible=True, interactive=False),                              # add_btn
                    gr.update(visible=False, interactive=False),                             # delete_btn
                    gr.update(visible=True),                                                 # cancel_edit_btn
                    gr.update(visible=True, interactive=False),                              # row_to_delete
                ]

            def lease_dropdown_change(selected_value, prop_id):
                if not selected_value:
                    return gr.update(), gr.update(), gr.update(choices=[], value=None)
                lease_id = selected_value.split("(")[-1].replace(")", "").strip()
                bay_list = get_available_bays(prop_id, lease_id)
                return (
                    gr.update(value=selected_value),     # edit_lease
                    gr.update(value=""),                 # edit_bay
                    gr.update(choices=bay_list, value=None, visible=True)
                )

            def apply_bay_selection(bay):
                return bay or ""

            def apply_aps_selection(ap_list):
                return ", ".join(ap_list) if ap_list else ""

            def apply_edit(
                name, mobile, country, car_a, car_b, lease_auto, bay_auto, aps,
                active_flag, afterhours_flag, allaps_flag, row_id, prop_id,
                lease_choice, bay_choice
            ):
                """
                Edit flow:
                  - DO NOT change TenantUserName / mobile / country / TenantUID here.
                  - Only update: LeaseID, BayAllocated, PedestrianAPs, CarLicenceA/B, Active, AfterHours, AllAPs.
                """

                # Keep name/mobile/country as-is in the DB; UI shows them as [auto] and non-interactive.

                car_a_db = _lic_or_null(car_a)
                car_b_db = _lic_or_null(car_b)

                lease_val = (lease_choice or lease_auto or "").strip()
                bay_val   = (bay_choice   or bay_auto   or "").strip()
                active_db = bool(active_flag)
                afterhours_db = bool(afterhours_flag)
                allaps_db = bool(allaps_flag)
                print("ap groups: ", aps)
                uid = _uid()
                conn = get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    '''
                    UPDATE tenantuser
                    SET "LeaseID"       = %s,
                        "BayAllocated"  = %s,
                        "PedestrianAPs" = %s,
                        "CarLicenceA"   = %s,
                        "CarLicenceB"   = %s,
                        "Active"        = %s,
                        "AfterHours"    = %s,
                        "AllAPs"        = %s
                    WHERE "ID" = %s AND "UserID" = %s AND "PropID" = %s
                    ''',
                    (
                        lease_val, bay_val, aps or "",
                        car_a_db, car_b_db,
                        active_db, afterhours_db, allaps_db,
                        int(row_id), uid, int(prop_id)
                    )
                )
                conn.commit()

                df, df_display = sync_tenantuser(prop_id)
                df = _normalize_tenantuser_columns(df)
                opts = get_row_options(df)
                conn.close()

                return (
                    gr.update(value=df_display, visible=True),                 # tenant_table
                    df, prop_id,                                               # state_df, state_prop_id
                    gr.update(choices=opts, value=None,
                              visible=True, interactive=True),                 # row_to_edit ACTIVE
                    gr.update(choices=opts, value=None,
                              visible=True, interactive=True),                 # row_to_delete ACTIVE
                    gr.update(visible=False),                                  # edit_group
                    gr.update(value=""),                                       # edit_name
                    gr.update(value=""),                                       # edit_mobile
                    gr.update(value=""),                                       # edit_country
                    gr.update(value=""),                                       # edit_car_a
                    gr.update(value=""),                                       # edit_car_b
                    gr.update(value="", interactive=False),                    # edit_lease
                    gr.update(value=""),                                       # edit_bay
                    gr.update(value=""),                                       # edit_aps
                    gr.update(value=False),                                    # edit_active
                    gr.update(value=False),                                    # edit_afterhours
                    gr.update(value=False),                                    # edit_allaps
                    gr.update(value="", visible=False),                        # hidden_id
                    gr.update(choices=[], visible=False),                      # bay_dropdown
                    gr.update(choices=[], visible=False),                      # aps_dropdown
                    gr.update(choices=[], visible=False),                      # lease_dropdown
                    gr.update(visible=False),                                  # apply_edit_btn
                    gr.update(visible=True, interactive=True),                 # add_btn ACTIVE
                    gr.update(visible=False)                                   # delete_btn
                )


            def start_new_permission(prop_id):
                """
                UI only:
                - Hide 'Select row to edit/delete'
                - Show Country/Mobile inputs
                - Show Process/Cancel New Permission buttons
                - Do NOT touch the database.
                """
                if not prop_id:
                    try:
                        gr.Warning("Select a property before adding a new Permission User.")
                    except Exception:
                        pass
                # If no prop_id, still return the correct number of outputs (keep UI unchanged)
                    return [gr.update() for _ in range(29)]

                prop_id_int = int(prop_id)
                df, df_display = sync_tenantuser(prop_id_int)
                opts = get_row_options(df)

                return [
                    gr.update(value=df_display, visible=True),            # tenant_table
                    df,                                                   # state_df
                    prop_id_int,                                          # state_prop_id
                    gr.update(choices=opts, value=None,
                              visible=False, interactive=False),          # row_to_edit HIDDEN
                    gr.update(choices=_get_leaseid_remove_options_for_prop(prop_id_int),
                              value=None,
                              visible=True, interactive=True),           # remove_lease_dropdown unchanged (visible)
                    gr.update(choices=opts, value=None,
                              visible=False, interactive=False),          # row_to_delete HIDDEN
                    gr.update(visible=False),                             # edit_group
                    gr.update(value=""),                                  # edit_name
                    gr.update(value=""),                                  # edit_mobile
                    gr.update(value=""),                                  # edit_country
                    gr.update(value=""),                                  # edit_car_a
                    gr.update(value=""),                                  # edit_car_b
                    gr.update(value="", interactive=False),               # edit_lease
                    gr.update(value=""),                                  # edit_bay
                    gr.update(value=""),                                  # edit_aps
                    gr.update(value=False),                               # edit_active
                    gr.update(value=False),                               # edit_afterhours
                    gr.update(value=False),                               # edit_allaps
                    gr.update(value="", visible=False),                   # hidden_id
                    gr.update(choices=[], visible=False),                 # bay_dropdown
                    gr.update(choices=[], visible=False),                 # aps_dropdown
                    gr.update(choices=[], visible=False),                 # lease_dropdown
                    gr.update(visible=False),                             # apply_edit_btn
                    gr.update(visible=True, interactive=False),           # add_btn DISABLED
                    gr.update(visible=False),                             # delete_btn
                    gr.update(value=DEFAULT_COUNTRY_LABEL,
                              interactive=True, visible=True),           # add_country VISIBLE
                    gr.update(value="", interactive=True, visible=True), # add_mobile VISIBLE
                    gr.update(visible=True, interactive=True),           # process_new_btn
                    gr.update(visible=True, interactive=True),           # cancel_new_btn
                ]

            def cancel_new_permission(prop_id):
                """
                UI only:
                - Hide Country/Mobile and Process/Cancel
                - Restore 'Select row to edit/delete' visible & active.
                """
                if not prop_id:
                    return [gr.update() for _ in range(29)]

                prop_id_int = int(prop_id)
                df, df_display = sync_tenantuser(prop_id_int)
                opts = get_row_options(df)
                lease_remove_opts = _get_leaseid_remove_options_for_prop(prop_id_int)

                return [
                    gr.update(value=df_display, visible=True),            # tenant_table
                    df,                                                   # state_df
                    prop_id_int,                                          # state_prop_id
                    gr.update(choices=opts, value=None,
                              visible=True, interactive=True),            # row_to_edit
                    gr.update(
                        choices=lease_remove_opts,
                        value=None,
                        visible=True,
                        interactive=True
                    ),                                                   # remove_lease_dropdown
                    gr.update(choices=opts, value=None,
                              visible=True, interactive=True),            # row_to_delete
                    gr.update(visible=False),                             # edit_group
                    gr.update(value=""),                                  # edit_name
                    gr.update(value=""),                                  # edit_mobile
                    gr.update(value=""),                                  # edit_country
                    gr.update(value=""),                                  # edit_car_a
                    gr.update(value=""),                                  # edit_car_b
                    gr.update(value="", interactive=False),               # edit_lease
                    gr.update(value=""),                                  # edit_bay
                    gr.update(value=""),                                  # edit_aps
                    gr.update(value=False),                               # edit_active
                    gr.update(value=False),                               # edit_afterhours
                    gr.update(value=False),                               # edit_allaps
                    gr.update(value="", visible=False),                   # hidden_id
                    gr.update(choices=[], visible=False),                 # bay_dropdown
                    gr.update(choices=[], visible=False),                 # aps_dropdown
                    gr.update(choices=[], visible=False),                 # lease_dropdown
                    gr.update(visible=False),                             # apply_edit_btn
                    gr.update(visible=True, interactive=True),            # add_btn
                    gr.update(visible=False),                             # delete_btn
                    gr.update(value=DEFAULT_COUNTRY_LABEL,
                              interactive=False, visible=False),          # add_country HIDDEN
                    gr.update(value="", interactive=False, visible=False),# add_mobile HIDDEN
                    gr.update(visible=False, interactive=False),          # process_new_btn
                    gr.update(visible=False, interactive=False),          # cancel_new_btn
                ]

            # ---------- NEW: enhanced create_row with mobile + country validation ----------


            def create_row(prop_id, new_mobile, new_country_label):
                """
                Process the new user's mobile + country:
                  - Validate number & country
                  - Build formatted full mobile
                  - Check user_details.phone
                  - Insert tenantuser row on success

                UI:
                  - While processing / on error: keep 'new permission mode'
                  - On success: restore normal mode (lists visible, add_* hidden)
                """
                # If somehow called without a property, just do nothing visual
                if not prop_id:
                    try:
                        gr.Warning("Select a property before processing a new Permission User.")
                    except Exception:
                        pass
                    return [gr.update() for _ in range(29)]

                prop_id_int = int(prop_id)
                df, df_display = sync_tenantuser(prop_id_int)

                def new_mode(country_label, mobile_val):
                    """New-permission mode: lists hidden, add_* visible, process/cancel visible."""
                    opts = get_row_options(df)
                    lease_remove_opts = _get_leaseid_remove_options_for_prop(prop_id_int)
                    return [
                        gr.update(value=df_display, visible=True),            # tenant_table
                        df,                                                   # state_df
                        prop_id_int,                                          # state_prop_id
                        gr.update(choices=opts, value=None,
                                  visible=False, interactive=False),          # row_to_edit
                        gr.update(
                            choices=lease_remove_opts,
                            value=None,
                            visible=True,
                            interactive=True
                        ),                                                   # remove_lease_dropdown
                        gr.update(choices=opts, value=None,
                                  visible=False, interactive=False),          # row_to_delete
                        gr.update(visible=False),                             # edit_group
                        gr.update(value=""),                                  # edit_name
                        gr.update(value=""),                                  # edit_mobile
                        gr.update(value=""),                                  # edit_country
                        gr.update(value=""),                                  # edit_car_a
                        gr.update(value=""),                                  # edit_car_b
                        gr.update(value="", interactive=False),               # edit_lease
                        gr.update(value=""),                                  # edit_bay
                        gr.update(value=""),                                  # edit_aps
                        gr.update(value=False),                               # edit_active
                        gr.update(value=False),                               # edit_afterhours
                        gr.update(value=False),                               # edit_allaps
                        gr.update(value="", visible=False),                   # hidden_id
                        gr.update(choices=[], visible=False),                 # bay_dropdown
                        gr.update(choices=[], visible=False),                 # aps_dropdown
                        gr.update(choices=[], visible=False),                 # lease_dropdown
                        gr.update(visible=False),                             # apply_edit_btn
                        gr.update(visible=True, interactive=False),           # add_btn
                        gr.update(visible=False),                             # delete_btn
                        gr.update(value=country_label or DEFAULT_COUNTRY_LABEL,
                                  interactive=True, visible=True),           # add_country
                        gr.update(value=mobile_val or "",
                                  interactive=True, visible=True),           # add_mobile
                        gr.update(visible=True, interactive=True),           # process_new_btn
                        gr.update(visible=True, interactive=True),           # cancel_new_btn
                    ]

                def normal_mode(df_norm, df_display_norm):
                    """Normal mode: lists visible, new-permission widgets hidden."""
                    opts = get_row_options(df_norm)
                    lease_remove_opts = _get_leaseid_remove_options_for_prop(prop_id_int)
                    return [
                        gr.update(value=df_display_norm, visible=True),       # tenant_table
                        df_norm,                                             # state_df
                        prop_id_int,                                         # state_prop_id
                        gr.update(choices=opts, value=None,
                                  visible=True, interactive=True),           # row_to_edit
                        gr.update(
                            choices=lease_remove_opts,
                            value=None,
                            visible=True,
                            interactive=True
                        ),                                                   # remove_lease_dropdown
                        gr.update(choices=opts, value=None,
                                  visible=True, interactive=True),           # row_to_delete
                        gr.update(visible=False),                            # edit_group
                        gr.update(value=""),                                 # edit_name
                        gr.update(value=""),                                 # edit_mobile
                        gr.update(value=""),                                 # edit_country
                        gr.update(value=""),                                 # edit_car_a
                        gr.update(value=""),                                 # edit_car_b
                        gr.update(value="", interactive=False),              # edit_lease
                        gr.update(value=""),                                 # edit_bay
                        gr.update(value=""),                                 # edit_aps
                        gr.update(value=False),                              # edit_active
                        gr.update(value=False),                              # edit_afterhours
                        gr.update(value=False),                              # edit_allaps
                        gr.update(value="", visible=False),                  # hidden_id
                        gr.update(choices=[], visible=False),                # bay_dropdown
                        gr.update(choices=[], visible=False),                # aps_dropdown
                        gr.update(choices=[], visible=False),                # lease_dropdown
                        gr.update(visible=False),                            # apply_edit_btn
                        gr.update(visible=True, interactive=True),           # add_btn
                        gr.update(visible=False),                            # delete_btn
                        gr.update(value=DEFAULT_COUNTRY_LABEL,
                                  interactive=False, visible=False),         # add_country
                        gr.update(value="", interactive=False, visible=False),# add_mobile
                        gr.update(visible=False, interactive=False),         # process_new_btn
                        gr.update(visible=False, interactive=False),         # cancel_new_btn
                    ]

                # --- Validation ---

                mobile_candidate = _mobile_to_db(new_mobile)
                if mobile_candidate is None or not is_valid_mobile(mobile_candidate):
                    try:
                        gr.Warning("Invalid mobile number. Please enter a valid 9-digit mobile (0XXXXXXXXX / 27XXXXXXXXX / +27XXXXXXXXX).")
                    except Exception:
                        pass
                    return new_mode(new_country_label, new_mobile)

                country_digits = _extract_country_digits(new_country_label) or ""
                if not country_digits:
                    try:
                        gr.Warning("Please select a valid country code.")
                    except Exception:
                        pass
                    return new_mode(new_country_label, new_mobile)

                raw_digits = re.sub(r"\D", "", str(new_mobile or "").strip())
                if raw_digits.startswith("27") and len(raw_digits) >= 11:
                    local_nine = raw_digits[-9:]
                elif raw_digits.startswith("0") and len(raw_digits) >= 10:
                    local_nine = raw_digits[-9:]
                else:
                    local_nine = raw_digits[-9:]

                if len(local_nine) != 9 or not local_nine.isdigit():
                    try:
                        gr.Warning("The mobile number must contain exactly 9 local digits after the leading 0.")
                    except Exception:
                        pass
                    return new_mode(new_country_label, new_mobile)

                tenant_full_mobile = f"+{country_digits}0{local_nine}"

                # Look up in user_details
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT "first_name", "last_name", "id" FROM user_details WHERE "phone" = %s LIMIT 1',
                    (tenant_full_mobile,)
                )
                hit = cursor.fetchone()

                if not hit:
                    try:
                        gr.Warning("This mobile number is not yet registered on the RyGo mobile app.")
                        gr.Warning("Ask the user to register in the app first, then try again.")
                    except Exception:
                        pass
                    conn.close()
                    return new_mode(new_country_label, new_mobile)

                first_name, last_name, tenant_uid = hit
                conn.close()

                tenant_name = f"{(first_name or '').strip()} {(last_name or '').strip()}".strip()
                if not tenant_name:
                    tenant_name = "<enter User details>"

                new_id = generate_id()

                # Insert new row
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO tenantuser ('
                    '  "ID","PropID","TenantUserName","TenantMobileNo","Country","TenantFullMobileNo","UserID",'
                    '  "LeaseID","BayAllocated","PedestrianAPs","CarLicenceA","CarLicenceB","Active","AfterHours","AllAPs","TenantUID"'
                    ') VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                    (
                        new_id,
                        prop_id_int,
                        tenant_name,         # TenantUserName
                        local_nine,          # TenantMobileNo (9-digit local)
                        country_digits,      # Country
                        tenant_full_mobile,  # TenantFullMobileNo
                        _uid(),              # UserID
                        None,                # LeaseID
                        "",                  # BayAllocated
                        "",                  # PedestrianAPs
                        None,                # CarLicenceA
                        None,                # CarLicenceB
                        True,                # Active
                        False,               # AfterHours
                        False,               # AllAPs (default False)
                        tenant_uid           # TenantUID
                    )
                )
                conn.commit()
                conn.close()

                # Reload and return to normal mode
                df_new, df_display_new = sync_tenantuser(prop_id_int)
                return normal_mode(df_new, df_display_new)


            def apply_delete(row_value, df, prop_id):
                # outputs: 24
                try:
                    # Helper: standard "reset" UI after delete or when nothing is selected
                    def _reset_ui(df_src, df_display_src):
                        opts = get_row_options(df_src)
                        return [
                            gr.update(value=df_display_src, visible=True),   # tenant_table
                            df_src,                                          # state_df
                            prop_id,                                         # state_prop_id
                            gr.update(choices=opts, value=None, visible=True, interactive=True),  # row_to_edit ACTIVE
                            gr.update(choices=opts, value=None, visible=True, interactive=True),  # row_to_delete ACTIVE
                            gr.update(visible=False),                        # edit_group
                            gr.update(value=""),                             # edit_name
                            gr.update(value=""),                             # edit_mobile
                            gr.update(value=""),                             # edit_country
                            gr.update(value=""),                             # edit_car_a
                            gr.update(value=""),                             # edit_car_b
                            gr.update(value="", interactive=False),          # edit_lease
                            gr.update(value=""),                             # edit_bay
                            gr.update(value=""),                             # edit_aps
                            gr.update(value=False),                          # edit_active
                            gr.update(value=False),                          # edit_afterhours
                            gr.update(value=False),                          # edit_allaps
                            gr.update(value="", visible=False),              # hidden_id
                            gr.update(choices=[], visible=False),            # bay_dropdown
                            gr.update(choices=[], visible=False),            # aps_dropdown
                            gr.update(choices=[], visible=False),            # lease_dropdown
                            gr.update(visible=False),                        # apply_edit_btn
                            gr.update(visible=True, interactive=True),       # add_btn ACTIVE
                            gr.update(visible=False),                        # delete_btn
                        ]

                    # 1) Nothing selected → just refresh & reset UI
                    if not row_value:
                        df_cur, df_display = sync_tenantuser(prop_id)
                        return _reset_ui(df_cur, df_display)

                    # 2) Find the row to delete using the display key
                    df_all = df.copy() if df is not None else pd.DataFrame()
                    if df_all.empty:
                        df_cur, df_display = sync_tenantuser(prop_id)
                        return _reset_ui(df_cur, df_display)

                    df_all["Display"] = _display_key_series(df_all)
                    row = df_all[df_all["Display"] == row_value]
                    if row.empty:
                        # Row not found anymore → refresh & reset
                        df_cur, df_display = sync_tenantuser(prop_id)
                        return _reset_ui(df_cur, df_display)

                    target_id = int(row.iloc[0]["ID"])

                    # 3) Delete from DB
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM tenantuser WHERE "ID" = %s', (target_id,))
                    conn.commit()
                    conn.close()

                    # 4) Reload and reset UI
                    df_new, df_display = sync_tenantuser(prop_id)
                    return _reset_ui(df_new, df_display)

                except Exception as e:
                    df_cur, df_display = sync_tenantuser(prop_id)
                    try:
                        gr.Error(f"Delete failed: {e}")
                    except Exception:
                        pass
                    return _reset_ui(df_cur, df_display)

            def cancel_delete(df=None, prop_id=None):
                # outputs: 24
                df_new, df_display = sync_tenantuser(prop_id)
                opts = get_row_options(df_new)
                return [
                    gr.update(value=df_display, visible=True),               # tenant_table
                    df_new,                                                 # state_df
                    prop_id,                                                # state_prop_id
                    gr.update(choices=opts, value=None, visible=True, interactive=True),   # row_to_edit ACTIVE
                    gr.update(choices=opts, value=None, visible=True, interactive=True),   # row_to_delete ACTIVE
                    gr.update(visible=False),                                # edit_group
                    gr.update(value=""),                                     # edit_name
                    gr.update(value=""),                                     # edit_mobile
                    gr.update(value=""),                                     # edit_country
                    gr.update(value=""),                                     # edit_car_a
                    gr.update(value=""),                                     # edit_car_b
                    gr.update(value="", interactive=False),                  # edit_lease
                    gr.update(value=""),                                     # edit_bay
                    gr.update(value=""),                                     # edit_aps
                    gr.update(value=False),                                  # edit_active
                    gr.update(value=False),                                  # edit_afterhours
                    gr.update(value=False),                                  # edit_allaps
                    gr.update(value="", visible=False),                      # hidden_id
                    gr.update(choices=[], visible=False),                    # bay_dropdown
                    gr.update(choices=[], visible=False),                    # aps_dropdown
                    gr.update(choices=[], visible=False),                    # lease_dropdown
                    gr.update(visible=False),                                # apply_edit_btn
                    gr.update(visible=True, interactive=True),               # add_btn ACTIVE
                    gr.update(visible=False),                                # delete_btn
                ]

            def cancel_edit(df, prop_id):
                df_new, df_display = sync_tenantuser(prop_id)
                opts = get_row_options(df_new)
                return [
                    df_display, df_new, prop_id,                        # tenant_table, state_df, state_prop_id
                    gr.update(choices=opts, value=None, visible=True, interactive=True),  # row_to_edit ACTIVE
                    gr.update(choices=opts, value=None, visible=True, interactive=True),  # row_to_delete ACTIVE
                    gr.update(visible=False),                           # edit_group
                    gr.update(value=""),                               # edit_name
                    gr.update(value=""),                               # edit_mobile
                    gr.update(value=""),                               # edit_country
                    gr.update(value=""),                               # edit_car_a
                    gr.update(value=""),                               # edit_car_b
                    gr.update(value="", interactive=False),            # edit_lease
                    gr.update(value=""),                               # edit_bay
                    gr.update(value=""),                               # edit_aps
                    gr.update(value=False),                            # edit_active
                    gr.update(value=False),                            # edit_afterhours
                    gr.update(value=False),                            # edit_allaps
                    gr.update(value="", visible=False),                # hidden_id
                    gr.update(choices=[], visible=False),              # bay_dropdown
                    gr.update(choices=[], visible=False),              # aps_dropdown
                    gr.update(choices=[], visible=False),              # lease_dropdown
                    gr.update(visible=False),                          # apply_edit_btn
                    gr.update(visible=True, interactive=True),         # add_btn ACTIVE
                    gr.update(visible=False),                          # delete_btn
                ]

            # ---------- NEW: "Remove Lease Permissions" handlers ----------

            def on_remove_lease_selection_change(selected_lease):
                """
                Toggle UI when a lease is chosen for removal.
                """
                if selected_lease:
                    # Enter "remove lease" mode: disable add/edit/delete selectors, show apply/cancel removal
                    return (
                        gr.update(visible=True, interactive=False),   # add_btn DISABLED
                        gr.update(visible=True, interactive=False),   # row_to_edit DISABLED
                        gr.update(visible=True, interactive=False),   # row_to_delete DISABLED
                        gr.update(visible=True),                      # apply_remove_lease_btn
                        gr.update(visible=True),                      # cancel_remove_lease_btn
                    )
                else:
                    # No lease selected: normal mode (buttons back to default)
                    return (
                        gr.update(visible=True, interactive=True),    # add_btn ACTIVE
                        gr.update(visible=True, interactive=True),    # row_to_edit
                        gr.update(visible=True, interactive=True),    # row_to_delete
                        gr.update(visible=False),                     # apply_remove_lease_btn
                        gr.update(visible=False),                     # cancel_remove_lease_btn
                    )

            def apply_remove_lease_permissions(selected_lease, prop_id):
                """
                Apply removal of permissions linked to a lease:
                  - Active = False, AfterHours = False
                  - LeaseID, BayAllocated, PedestrianAPs = NULL
                Then reset UI & refresh table and lease list.
                """
                if not prop_id:
                    # Nothing to do visually if no property
                    return [gr.update() for _ in range(27)]

                prop_id_int = int(prop_id)

                if selected_lease:
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        '''
                        UPDATE tenantuser
                        SET "Active" = FALSE,
                            "AfterHours" = FALSE,
                            "LeaseID" = NULL,
                            "BayAllocated" = NULL,
                            "PedestrianAPs" = NULL
                        WHERE "UserID" = %s AND "PropID" = %s AND "LeaseID" = %s
                        ''',
                        (_uid(), prop_id_int, selected_lease)
                    )
                    conn.commit()
                    conn.close()

                # Refresh after update
                df_new, df_display = sync_tenantuser(prop_id_int)
                opts = get_row_options(df_new)
                lease_remove_opts = _get_leaseid_remove_options_for_prop(prop_id_int)

                return [
                    gr.update(value=df_display, visible=True),                # tenant_table
                    df_new,                                                   # state_df
                    prop_id_int,                                              # state_prop_id
                    gr.update(choices=opts, value=None,
                              visible=True, interactive=True),                # row_to_edit
                    gr.update(choices=opts, value=None,
                              visible=True, interactive=True),                # row_to_delete
                    gr.update(visible=False),                                 # edit_group
                    gr.update(value=""),                                      # edit_name
                    gr.update(value=""),                                      # edit_mobile
                    gr.update(value=""),                                      # edit_country
                    gr.update(value=""),                                      # edit_car_a
                    gr.update(value=""),                                      # edit_car_b
                    gr.update(value="", interactive=False),                   # edit_lease
                    gr.update(value=""),                                      # edit_bay
                    gr.update(value=""),                                      # edit_aps
                    gr.update(value=False),                                   # edit_active
                    gr.update(value=False),                                   # edit_afterhours
                    gr.update(value=False),                                   # edit_allaps
                    gr.update(value="", visible=False),                       # hidden_id
                    gr.update(choices=[], visible=False),                     # bay_dropdown
                    gr.update(choices=[], visible=False),                     # aps_dropdown
                    gr.update(choices=[], visible=False),                     # lease_dropdown
                    gr.update(visible=False),                                 # apply_edit_btn
                    gr.update(visible=True, interactive=True),                # add_btn
                    gr.update(visible=False),                                 # delete_btn
                    gr.update(
                        choices=lease_remove_opts,
                        value=None,
                        visible=True,
                        interactive=True
                    ),                                                        # remove_lease_dropdown
                    gr.update(visible=False),                                 # apply_remove_lease_btn
                    gr.update(visible=False),                                 # cancel_remove_lease_btn
                ]

            def cancel_remove_lease_permissions(prop_id):
                """
                Cancel lease-permission removal:
                  - No DB changes
                  - Reset UI & refresh table and lease list
                """
                if not prop_id:
                    return [gr.update() for _ in range(27)]

                prop_id_int = int(prop_id)
                df_new, df_display = sync_tenantuser(prop_id_int)
                opts = get_row_options(df_new)
                lease_remove_opts = _get_leaseid_remove_options_for_prop(prop_id_int)

                return [
                    gr.update(value=df_display, visible=True),                # tenant_table
                    df_new,                                                   # state_df
                    prop_id_int,                                              # state_prop_id
                    gr.update(choices=opts, value=None,
                              visible=True, interactive=True),                # row_to_edit
                    gr.update(choices=opts, value=None,
                              visible=True, interactive=True),                # row_to_delete
                    gr.update(visible=False),                                 # edit_group
                    gr.update(value=""),                                      # edit_name
                    gr.update(value=""),                                      # edit_mobile
                    gr.update(value=""),                                      # edit_country
                    gr.update(value=""),                                      # edit_car_a
                    gr.update(value=""),                                      # edit_car_b
                    gr.update(value="", interactive=False),                   # edit_lease
                    gr.update(value=""),                                      # edit_bay
                    gr.update(value=""),                                      # edit_aps
                    gr.update(value=False),                                   # edit_active
                    gr.update(value=False),                                   # edit_afterhours
                    gr.update(value=False),                                   # edit_allaps
                    gr.update(value="", visible=False),                       # hidden_id
                    gr.update(choices=[], visible=False),                     # bay_dropdown
                    gr.update(choices=[], visible=False),                     # aps_dropdown
                    gr.update(choices=[], visible=False),                     # lease_dropdown
                    gr.update(visible=False),                                 # apply_edit_btn
                    gr.update(visible=True, interactive=True),                # add_btn
                    gr.update(visible=False),                                 # delete_btn
                    gr.update(
                        choices=lease_remove_opts,
                        value=None,
                        visible=True,
                        interactive=True
                    ),                                                        # remove_lease_dropdown
                    gr.update(visible=False),                                 # apply_remove_lease_btn
                    gr.update(visible=False),                                 # cancel_remove_lease_btn
                ]

            # ---------- WIRING ----------

            prop_map = {}
            print("DEBUG (prime): starting with UID =", _uid())
            
            # Start empty; let demo.load populate it
            prop_dropdown = gr.Dropdown(label="Select Property (Owner)", choices=[], value=None)
            
            btn_reload_props = gr.Button("Reload Properties", variant="secondary")
            btn_reload_props.click(_prime_uid_then_fill_properties, outputs=[prop_dropdown])

            tenant_table = gr.Dataframe(
                label="List of Users with Permissions to Restricted Access Points",
                visible=False,
                interactive=False
            )           

            state_df = gr.State()
            state_prop_id = gr.State()

            # Populate on load
            demo.load(_prime_uid_then_fill_properties, inputs=[], outputs=[prop_dropdown])

            with gr.Row():
                # REMOVED: edit_btn
                add_btn = gr.Button("Add New User", interactive=False)
                delete_btn = gr.Button("Delete", visible=False)
                process_new_btn = gr.Button("Process New User", visible=False, interactive=False)
                cancel_new_btn = gr.Button("Cancel New User", visible=False, interactive=False)

            # NEW: Inputs used when creating a new permission row
            with gr.Row():
                add_country = gr.Dropdown(
                    label="Country Code for New User",
                    choices=COUNTRY_CHOICES,
                    value=DEFAULT_COUNTRY_LABEL,
                    interactive=False,
                    visible=False       # ← start hidden
                )
                add_mobile = gr.Textbox(
                    label="Mobile No of New User (9 digits)",
                    interactive=False,
                    visible=False       # ← start hidden
                )

            with gr.Row():
                row_to_edit = gr.Dropdown(
                    label="Select User to edit",
                    choices=[],
                    interactive=False,
                    value=None
                )
                # NEW: Remove Lease Permissions dropdown
                remove_lease_dropdown = gr.Dropdown(
                    label="Remove Lease Permissions",
                    choices=[],
                    interactive=False,
                    value=None
                )
                row_to_delete = gr.Dropdown(
                    label="Select User to delete",
                    choices=[],
                    interactive=False,
                    value=None
                )

            with gr.Row():
                apply_delete_btn = gr.Button("Apply Delete", visible=False)
                cancel_delete_btn = gr.Button("Cancel Delete", visible=False)

            # NEW: buttons for Remove Lease Permissions
            with gr.Row():
                apply_remove_lease_btn = gr.Button(
                    "Apply removal of permissions linked to lease",
                    visible=False
                )
                cancel_remove_lease_btn = gr.Button(
                    "Cancel removal of permissions linked to lease",
                    visible=False
                )

            with gr.Group(visible=False) as edit_group:
                gr.Markdown(
                    """<div style="background-color:#666666; color:#FFFFFF; padding:10px; border-radius:0px;">Input / Edit User's information and permissions</div>"""
                )

                with gr.Row():
                    lease_dropdown = gr.Dropdown(label="Select Lease/Permission Category connected to User", choices=[], visible=False)
                    bay_dropdown = gr.Dropdown(label="Select Parking Bay allocated to Lease/Permission Category", choices=[], visible=False, allow_custom_value=True)
                    aps_dropdown = gr.Dropdown(label="Select Non-Parking Bay AP-Groups", choices=[], visible=False, multiselect=True)

                with gr.Row():
                    edit_car_a = gr.Textbox(label="User's Car Licence A")
                    edit_car_b = gr.Textbox(label="User's Car Licence B")
                    edit_active = gr.Checkbox(label="Tick to Activate User", value=True)
                    edit_afterhours = gr.Checkbox(label="Tick for After-Hours Access", value=False)
                    edit_allaps = gr.Checkbox(label="Tick to allow ALL AP Groups", value=False)

                gr.Markdown(
                    """<div style="background-color:#666666; color:#FFFFFF; padding:10px; border-radius:0px;">Updated User's Details</div>"""
                )

                with gr.Row():
                    edit_lease = gr.Textbox(label="Lease/Permission Category", interactive=False)
                    edit_name = gr.Textbox(label="Name of User", interactive=False)
                    edit_country = gr.Textbox(label="Country Code", interactive=False)
                    edit_mobile = gr.Textbox(label="Mobile No of User", interactive=False)


                with gr.Row():
                    edit_bay = gr.Textbox(label="Bay Allocated to User", interactive=False)
                    edit_aps = gr.Textbox(label="User's AP-Groups", interactive=False)

                with gr.Row():
                    hidden_id = gr.Textbox(visible=False)

                with gr.Row():
                    apply_edit_btn = gr.Button("Apply Edit", visible=False)
                    cancel_edit_btn = gr.Button("Cancel Edit", visible=False)

            # property change

            prop_dropdown.change(
                select_property,
                inputs=prop_dropdown,
                outputs=[
                    tenant_table, state_df, state_prop_id,
                    row_to_edit, remove_lease_dropdown, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country, edit_car_a, edit_car_b,
                    edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown,
                    apply_edit_btn, add_btn, delete_btn,
                    add_country, add_mobile, process_new_btn, cancel_new_btn
                ]
            )


            # edit selection
            row_to_edit.change(
                select_row,
                inputs=[row_to_edit, state_df, state_prop_id],
                outputs=[
                    edit_name, edit_mobile, edit_country,
                    edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown,
                    edit_group, apply_edit_btn, add_btn, delete_btn, cancel_edit_btn, row_to_delete
                ]
            )

            lease_dropdown.change(
                lease_dropdown_change,
                inputs=[lease_dropdown, state_prop_id],
                outputs=[edit_lease, edit_bay, bay_dropdown]
            )

            bay_dropdown.change(apply_bay_selection, inputs=bay_dropdown, outputs=edit_bay)
            aps_dropdown.change(apply_aps_selection, inputs=aps_dropdown, outputs=edit_aps)

            # apply edit
            apply_edit_btn.click(
                apply_edit,
                inputs=[
                    edit_name, edit_mobile, edit_country,
                    edit_car_a, edit_car_b,
                    edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id, state_prop_id,
                    lease_dropdown, bay_dropdown
                ],
                outputs=[
                    tenant_table, state_df, state_prop_id, row_to_edit, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country,
                    edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown, apply_edit_btn, add_btn, delete_btn
                ]
            )

            add_btn.click(
                start_new_permission,
                inputs=[state_prop_id],
                outputs=[
                    tenant_table, state_df, state_prop_id,
                    row_to_edit, remove_lease_dropdown, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country,
                    edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown, apply_edit_btn, add_btn, delete_btn,
                    add_country, add_mobile, process_new_btn, cancel_new_btn
                ]
            )

            process_new_btn.click(
                create_row,
                inputs=[state_prop_id, add_mobile, add_country],
                outputs=[
                    tenant_table, state_df, state_prop_id,
                    row_to_edit, remove_lease_dropdown, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country,
                    edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown, apply_edit_btn, add_btn, delete_btn,
                    add_country, add_mobile, process_new_btn, cancel_new_btn
                ]
            )

            cancel_new_btn.click(
                cancel_new_permission,
                inputs=[state_prop_id],
                outputs=[
                    tenant_table, state_df, state_prop_id,
                    row_to_edit, remove_lease_dropdown, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country,
                    edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown, apply_edit_btn, add_btn, delete_btn,
                    add_country, add_mobile, process_new_btn, cancel_new_btn
                ]
            )



            # delete selection toggles (NOTE: edit_btn removed)
            row_to_delete.change(
                lambda row: (
                    gr.update(visible=True, interactive=False),             # add_btn
                    gr.update(visible=False, interactive=False),            # delete_btn (keep hidden in delete mode)
                    gr.update(visible=True),                                # apply_delete_btn
                    gr.update(visible=True),                                # cancel_delete_btn
                    gr.update(visible=True, interactive=False)              # row_to_edit: visible BUT not active while deleting
                ) if row else (
                    gr.update(visible=True, interactive=True),              # add_btn
                    gr.update(visible=False, interactive=False),            # delete_btn
                    gr.update(visible=False),                               # apply_delete_btn
                    gr.update(visible=False),                               # cancel_delete_btn
                    gr.update(visible=True, interactive=True)               # row_to_edit: visible AND active when not deleting
                ),
                inputs=row_to_delete,
                outputs=[add_btn, delete_btn, apply_delete_btn, cancel_delete_btn, row_to_edit]
            )

            # apply delete
            apply_delete_btn.click(
                apply_delete,
                inputs=[row_to_delete, state_df, state_prop_id],
                outputs=[
                    tenant_table, state_df, state_prop_id, row_to_edit, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country, edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown, apply_edit_btn, add_btn, delete_btn
                ]
            )

            # cancel delete
            cancel_delete_btn.click(
                cancel_delete,
                inputs=[state_df, state_prop_id],
                outputs=[
                    tenant_table, state_df, state_prop_id, row_to_edit, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country, edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown, apply_edit_btn, add_btn, delete_btn
                ]
            )

            # cancel edit
            cancel_edit_btn.click(
                cancel_edit,
                inputs=[state_df, state_prop_id],
                outputs=[
                    tenant_table, state_df, state_prop_id, row_to_edit, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country, edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown, apply_edit_btn, add_btn, delete_btn
                ]
            )

            # NEW: Remove Lease Permissions wiring

            remove_lease_dropdown.change(
                on_remove_lease_selection_change,
                inputs=remove_lease_dropdown,
                outputs=[add_btn, row_to_edit, row_to_delete, apply_remove_lease_btn, cancel_remove_lease_btn]
            )

            apply_remove_lease_btn.click(
                apply_remove_lease_permissions,
                inputs=[remove_lease_dropdown, state_prop_id],
                outputs=[
                    tenant_table, state_df, state_prop_id, row_to_edit, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country, edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown, apply_edit_btn, add_btn, delete_btn,
                    remove_lease_dropdown, apply_remove_lease_btn, cancel_remove_lease_btn
                ]
            )

            cancel_remove_lease_btn.click(
                cancel_remove_lease_permissions,
                inputs=[state_prop_id],
                outputs=[
                    tenant_table, state_df, state_prop_id, row_to_edit, row_to_delete, edit_group,
                    edit_name, edit_mobile, edit_country, edit_car_a, edit_car_b, edit_lease, edit_bay, edit_aps,
                    edit_active, edit_afterhours, edit_allaps, hidden_id,
                    bay_dropdown, aps_dropdown, lease_dropdown, apply_edit_btn, add_btn, delete_btn,
                    remove_lease_dropdown, apply_remove_lease_btn, cancel_remove_lease_btn
                ]
            )

    return demo
