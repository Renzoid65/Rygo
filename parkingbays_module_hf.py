from pathlib import Path
import gradio as gr
import psycopg2
from psycopg2 import sql
import pandas as pd
import random
import re
import os
import json
import ast
import base64
import hashlib
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet, InvalidToken  # not just Fernet
import time





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
        import pickle
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



def launch_parkingbays_module(get_user_id=None):

    if _coerce_uid(ActiveUserID):
        return ActiveUserID

    
    # ========== UTILS ==========
    def generate_id():
        return random.randint(111111111111111, 999999999999999)

    # ---------- Data access helpers (use _current_uid on every call) ----------

    def load_property_list():
        uid = _uid()
        print(f"DEBUG load_property(): ActiveUserID={uid}")
        if uid is None:
            # Return empty to avoid crashing the UI; user can press “Check UID”
            return pd.DataFrame(columns=["ID","PropID","Name","UserID","Owner","TotalBays","Active"])
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            'SELECT "ID","PropID","Name","UserID","Owner","TotalBays","Active" '
            'FROM property WHERE "UserID" = %s AND "Active" = TRUE',
            (uid,)
        )
        rows = cur.fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=["ID","PropID","Name","UserID","Owner","TotalBays","Active"])

    def load_parking(prop_id):
        uid = _uid()
        if uid is None:
            return pd.DataFrame(columns=["ID","PropID","Name","UserID","BayType","BayNo","TenantName","LeaseEnd","LeaseID","BayAccessNo"])
        conn = get_connection()
        df = pd.read_sql(
            'SELECT "ID","PropID","Name","UserID","BayType","BayNo","TenantName","LeaseEnd","LeaseID","BayAccessNo" '
            'FROM parkingbays WHERE "UserID" = %s AND "PropID" = %s',
            conn, params=(uid, prop_id)
        )
        conn.close()
        return df

    def sync_parking(df, prop_id):
        uid = _uid()
        if uid is None:
            return load_parking(prop_id)
    
        conn = get_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM parkingbays WHERE "UserID" = %s AND "PropID" = %s', (uid, prop_id))
    
        def clean(v):
            if pd.isna(v) or v == "":
                return None
            return v
    
        cols = ["ID","PropID","Name","UserID","BayType","BayNo","TenantName","LeaseEnd","LeaseID","BayAccessNo"]
        for _, row in df.iterrows():
            values = tuple(clean(row.get(c)) for c in cols)
            cur.execute('INSERT INTO parkingbays VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', values)
    
        conn.commit()
        conn.close()
        return load_parking(prop_id)


    def _get_user_company_id():
        uid = _uid()
        if uid is None:
            return None
        conn = get_connection()
        cur = conn.cursor()
        cur.execute('SELECT "UserCompanyID" FROM managerusers WHERE "MUserID" = %s', (uid,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else None

    parking_columns = ["ID", "PropID", "Name", "UserID", "BayType", "BayNo", "TenantName", "LeaseEnd", "LeaseID", "BayAccessNo",]
    visible_columns = ["BayNo", "BayType", "LeaseID", "TenantName", "BayAccessNo"]
    display_labels = {"BayType":"Bay Type", "BayNo":"Bay No", "LeaseID":"Unit / Lease ID", "TenantName":"Tenant/Person's Name","BayAccessNo":"AP-Group(s) Allocated",}    


    with gr.Blocks() as demo:

        with gr.Tab("Manage Parking Bays, connect bays to access points and to tenants"):

            def _uid() -> Optional[int]:
                # convenience wrapper used throughout the module
                return resolve_and_cache_uid(get_user_id)

            # ----- Debug: UID + DB connectivity -----
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
                return f"**ActiveUserID**: `{uid}`  ·  **DB**: {conn_txt}"

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

            # ========== APP ==========
    
            def clean_value(val):
                if pd.isna(val) or val == "":
                    return None
                return val

            # ---- Helpers / cascades ----

            def _apply_ordering(df: pd.DataFrame, sort_choice: str) -> pd.DataFrame:
                """
                Sort df by one of: 'Bay No', 'Tenant Name', 'Unit ID'.
                - Bay No: natural-ish (numbers inside strings sorted numerically)
                - Text columns: case-insensitive
                """
                if df is None or df.empty:
                    return df
            
                choice_to_col = {
                    "Bay No": "BayNo",
                    "Tenant Name": "TenantName",
                    "Unit ID": "LeaseID",
                }
                col = choice_to_col.get((sort_choice or "Bay No").strip(), "BayNo")
                if col not in df.columns:
                    return df
            
                if col == "BayNo":
                    # natural-ish sort: split into digits/non-digits
                    def _natkey(x):
                        s = "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
                        parts = re.findall(r"\d+|\D+", s)
                        return [int(p) if p.isdigit() else p.lower() for p in parts]
                    return df.sort_values(by=col, key=lambda s: s.map(_natkey), kind="mergesort", na_position="last").reset_index(drop=True)
                else:
                    return df.sort_values(by=col, key=lambda s: s.astype(str).str.casefold(), kind="mergesort", na_position="last").reset_index(drop=True)


            def _update_tenantuser_bayallocated_on_bayno_change(user_id: int, prop_id: int, bay_id: int, old_bayno, new_bayno) -> int:
                """
                If a parking bay's BayNo changes, update tenantuser."BayAllocated" accordingly.
                Handles both "BayNo (ID)" and plain "BayNo" formats.
                Returns total rows updated.
                """
                old_s = ("" if old_bayno is None else str(old_bayno)).strip()
                new_s = ("" if new_bayno is None else str(new_bayno)).strip()

                if not old_s or not new_s or old_s == new_s:
                    return 0

                combo_old = f"{old_s} ({bay_id})"
                combo_new = f"{new_s} ({bay_id})"

                conn = get_connection()
                cur = conn.cursor()
                total = 0
                try:
                    # Update the "BayNo (ID)" format
                    cur.execute(
                        'UPDATE tenantuser SET "BayAllocated" = %s '
                        'WHERE "UserID" = %s AND "PropID" = %s AND btrim("BayAllocated") = btrim(%s)',
                        (combo_new, user_id, prop_id, combo_old)
                    )
                    total += cur.rowcount

                    # Update the plain "BayNo" format as a fallback
                    cur.execute(
                        'UPDATE tenantuser SET "BayAllocated" = %s '
                        'WHERE "UserID" = %s AND "PropID" = %s AND btrim("BayAllocated") = btrim(%s)',
                        (new_s, user_id, prop_id, old_s)
                    )
                    total += cur.rowcount

                    conn.commit()
                    print(f"[DEBUG] Updated tenantuser.BayAllocated rows: {total} (BayID={bay_id}, {old_s}→{new_s})")
                except Exception as e:
                    conn.rollback()
                    print(f"❌ Error updating tenantuser.BayAllocated for bay change: {e}")
                finally:
                    conn.close()
                return total

            I_TN = 4
            I_LI = 3
            
            def _norm_str(x): return (str(x).strip() if x is not None else "")
            
            def _maybe_show_lease_menu(*vals):
                """
                Decide whether to run a simple single-row edit (apply_edits)
                or the global 'Option 1' behaviour (apply_edits_with_choice).

                If TenantName or LeaseID changed AND there are linked TenantUser rows
                using the old combined LeaseID, we automatically run the global
                behaviour (Option 1) with no menu shown.
                """
                try:
                    *edit_vals, prop_id, edit_row_id = vals
                except Exception as e:
                    print("❌ _maybe_show_lease_menu unpack error:", e)
                    # Let apply_edits handle the error UI
                    return apply_edits(*vals, "")

                # If we don't have a property / row, just let apply_edits show its own error
                if not prop_id or not edit_row_id:
                    return apply_edits(*vals, "")

                try:
                    prop_id = int(prop_id)
                    edit_row_id = int(edit_row_id)
                except Exception:
                    return apply_edits(*vals, "")

                # New (edited) values from UI
                new_tn = _norm_str(edit_vals[tenantname_idx] if tenantname_idx < len(edit_vals) else "")
                new_li = _norm_str(edit_vals[leaseid_idx]    if leaseid_idx    < len(edit_vals) else "")

                # BEFORE values from DB
                tn_before = li_before = ""
                conn = get_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            'SELECT COALESCE("TenantName", \'\'), COALESCE("LeaseID", \'\') '
                            'FROM parkingbays WHERE "ID" = %s',
                            (edit_row_id,)
                        )
                        row = cur.fetchone()
                        if row:
                            tn_before = _norm_str(row[0])
                            li_before = _norm_str(row[1])
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

                print("_maybe_show_lease_menu:")
                print("row: ", edit_row_id)
                print("tn_before: ", tn_before, "li_before: ", li_before)
                print("new_tn: ", new_tn, "new_li: ", new_li)

                tenant_changed = (tn_before != new_tn)
                lease_changed  = (li_before != new_li)

                # If nothing relevant changed, just do a simple apply_edits
                if not (tenant_changed or lease_changed):
                    return apply_edits(*vals, "")

                # Check if there are any linked TenantUser rows for the old combined LeaseID
                before_combined = f"{tn_before} ({li_before})".strip()
                needs_global = False
                if before_combined:
                    uid = _uid()
                    if uid is not None:
                        conn_chk = get_connection()
                        try:
                            cur_chk = conn_chk.cursor()
                            cur_chk.execute(
                                'SELECT COUNT(*) FROM tenantuser '
                                'WHERE "UserID" = %s AND "PropID" = %s AND btrim("LeaseID") = btrim(%s)',
                                (uid, prop_id, before_combined),
                            )
                            rows_count = cur_chk.fetchone()[0]
                            needs_global = rows_count > 0
                            print("Linked tenantuser rows for old LeaseID:", rows_count)
                        finally:
                            conn_chk.close()

                if needs_global:
                    # Run Option 1 behaviour directly (global update)
                    forced_label = "Option 1: automatic global update"
                    return apply_edits_with_choice(*vals, forced_label)

                # No linked tenantusers → simple one-row edit
                return apply_edits(*vals, "")


            # --- NEW: APGroups parser ---------------------------------------
            def _parse_apgroups_text(raw):
                """
                Parse property.APGroups text into a list of (group_name, aps_string) tuples.
                Accepts Python-literal or list/tuple-like structures.
                Example stored format:
                    [
                        ("Front Gates", "Gate A (123), Gate B (456)"),
                        ("Basement Doors", "Door 1 (789)")
                    ]
                Returns: [("Front Gates", "..."), ("Basement Doors", "..."), ...]
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



            def _needs_options_menu(old_lease: str | None, old_tenant: str | None,
                                    new_lease: str | None, new_tenant: str | None) -> bool:
                """Return True if either LeaseID or TenantName changed."""
                def _norm(s): return ("" if s is None else str(s).strip())
                return _norm(old_lease) != _norm(new_lease) or _norm(old_tenant) != _norm(new_tenant)

            def _get_table_columns(cursor, table_name: str) -> set[str]:
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema='public' AND table_name=%s
                    """,
                    (table_name,),
                )
                return {r[0] for r in cursor.fetchall()}

            def _apply_choice_parking_and_tenantuser(df, prop_id, before_combined, after_combined, choice_label):
                """
                In ONE transaction:
                  - Replace parkingbays for prop_id with `df`
                  - Update tenantuser LeaseID (combined) wherever it matches the *starting*
                    combined LeaseID (before_combined), using Option 1 semantics
                  - For ALL cases:
                      (a) if TenantName == starting tenant -> set to new tenant (case-insensitive)
                      (b) if LeaseID    == starting lease  -> set to new lease (case-insensitive, punctuation-agnostic)
                Returns: refreshed parking DataFrame for prop_id
                """
                # ---- Normalize inputs ----
                prop_id = int(prop_id)
                before_combined = (before_combined or "").strip()
                after_combined  = (after_combined or "").strip()
                print("before edit: ", before_combined, "  After edit", after_combined, "option (ignored): ", choice_label)
                
                # Local helper: split "Tenant Name (LeaseID)" → (tenant_name, lease_id)
                def _split_combined_tenant_lease(combined: str) -> tuple[str, str]:
                    s = (combined or "").strip()
                    if not s:
                        return "", ""
                    if "(" in s and s.endswith(")"):
                        tn = s[: s.rfind("(")].strip()
                        li = s[s.rfind("(") + 1 : -1].strip()
                        return tn, li
                    return s, ""

                uid = _uid()
                conn = get_connection()
                try:
                    cursor = conn.cursor()

                    # ---- Replace parkingbays for this property (DELETE + INSERT) ----
                    cursor.execute(
                        'DELETE FROM parkingbays WHERE "UserID" = %s AND "PropID" = %s',
                        (uid, prop_id)
                    )

                    def clean_value_local(v):
                        if pd.isna(v) or v == "":
                            return None
                        return v

                    for _, row in df.iterrows():
                        values = tuple(clean_value_local(row.get(col, None)) for col in parking_columns)
                        cursor.execute(
                            "INSERT INTO parkingbays VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                            values
                        )

                    # ---- Cascade only if the combined text actually changed ----
                    if before_combined and after_combined and before_combined != after_combined:
                        # Look for matching tenantuser rows on the *starting* combined LeaseID
                        sel_sql = (
                            'SELECT "ID" FROM tenantuser '
                            'WHERE "UserID" = %s AND "PropID" = %s AND btrim("LeaseID") = btrim(%s)'
                        )
                        cursor.execute(sel_sql, (uid, prop_id, before_combined))
                        tu_rows = cursor.fetchall()
                        print("Rows in tenantuser to change: ", tu_rows)

                        # TenantUser updates: always Option 1 semantics now
                        if tu_rows:
                            upd_tu = (
                                'UPDATE tenantuser SET "LeaseID" = %s '
                                'WHERE "UserID" = %s AND "PropID" = %s AND btrim("LeaseID") = btrim(%s)'
                            )
                            cursor.execute(upd_tu, (after_combined, uid, prop_id, before_combined))

                        # ===== Global updates in parkingbays =====
                        start_tn, start_li = _split_combined_tenant_lease(before_combined)
                        end_tn,   end_li   = _split_combined_tenant_lease(after_combined)

                        # (a) TenantName global update (case-insensitive compare)
                        if start_tn and end_tn and start_tn != end_tn:
                            upd_pb_tn = (
                                'UPDATE parkingbays SET "TenantName" = %s '
                                'WHERE "UserID" = %s AND "PropID" = %s '
                                'AND lower(btrim("TenantName")) = lower(btrim(%s))'
                            )
                            cursor.execute(upd_pb_tn, (end_tn, uid, prop_id, start_tn))

                        # (b) LeaseID global update (case-insensitive + punctuation-agnostic)
                        if start_li and end_li and start_li != end_li:
                            upd_pb_li = (
                                'UPDATE parkingbays SET "LeaseID" = %s '
                                'WHERE "UserID" = %s AND "PropID" = %s '
                                'AND lower(regexp_replace(btrim("LeaseID"), \'[^[:alnum:]]+\', \'\', \'g\')) = '
                                '    lower(regexp_replace(btrim(%s),        \'[^[:alnum:]]+\', \'\', \'g\'))'
                            )
                            cursor.execute(upd_pb_li, (end_li, uid, prop_id, start_li))

                    conn.commit()

                except Exception as e:
                    conn.rollback()
                    print("❌ Transaction failed (apply choice):", e)
                    raise
                finally:
                    conn.close()

                # Return refreshed parking dataframe
                return load_parking(prop_id)


            # ----------- PROPERTY SELECTED LOGIC -----------

            def on_property_selected(label, order_choice):
                if label in prop_map:
                    pid, pname = prop_map[label]
                    uid = _uid()
                    if uid is not None:
                        try:
                            conn_sync = get_connection()
                            cur_sync = conn_sync.cursor()
                            cur_sync.execute(
                                'UPDATE parkingbays SET "Name" = %s '
                                'WHERE "UserID" = %s AND "PropID" = %s AND btrim("Name") <> btrim(%s)',
                                (pname, uid, pid, pname)
                            )
                            conn_sync.commit()
                        except Exception:
                            try: conn_sync.rollback()
                            except: pass
                        finally:
                            try: conn_sync.close()
                            except: pass

                        try:
                            conn_ap = get_connection()
                            cur_ap = conn_ap.cursor()
                            cur_ap.execute(
                                'SELECT "APGroups" FROM property WHERE "PropID" = %s AND "UserID" = %s',
                                (int(pid), uid)
                            )
                                # uid may be None if somehow missing, but then above code wouldn't run
                            row_ap = cur_ap.fetchone()
                            raw_apgroups = row_ap[0] if row_ap and row_ap[0] is not None else None
                            groups = _parse_apgroups_text(raw_apgroups)
                            conn_ap.close()
                            apgroups_by_prop[int(pid)] = groups
                            print(f"DEBUG APGroups for PropID={pid}: {groups}")
                        except Exception as e:
                            print(f"⚠️ Failed to load APGroups for PropID={pid}: {e}")
                            apgroups_by_prop[int(pid)] = []
            
                    df = load_parking(pid)
                    df = _apply_ordering(df, order_choice or "Bay No")  # ← apply ordering here
                    # NEW: load APGroups for this property and cache it

            
                    row_labels = [f"{row['BayNo'] or '-'} ({row['BayType'] or '-'})" for _, row in df.iterrows()]
                    id_map = {lbl: row["ID"] for lbl, row in zip(row_labels, df.to_dict("records"))}
                    df_to_show = df[visible_columns].rename(columns=display_labels)
                    return (
                        gr.update(value=df_to_show, visible=True),
                        pid, pname,
                        gr.update(choices=row_labels, visible=True, value=None, interactive=not df.empty),
                        gr.update(visible=True, interactive=True),
                        gr.update(choices=row_labels, visible=True, value=None, interactive=not df.empty),
                        id_map,
                        gr.update(visible=False),
                        gr.update(visible=True),
                    )
            
                # none selected
                return (
                    gr.update(value=pd.DataFrame(columns=[display_labels.get(c, c) for c in visible_columns]), visible=False),
                    None, None,
                    gr.update(choices=[], visible=False, value=None),
                    gr.update(visible=False, interactive=False),
                    gr.update(choices=[], visible=False, value=None),
                    {},
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            
            # ----------- EDIT ROW -----------
            def on_row_selected(label, df_unused, prop_id, id_map):

                if not label or label not in id_map:
                    return (
                        *[gr.update() for _ in edit_boxes],     # 5
                        gr.update(),                            # bay_access_dropdown
                        gr.update(visible=False),               # edit_group
                        gr.update(visible=False),               # apply_btn
                        gr.update(visible=False),               # cancel_edit_btn
                        None,                                   # selected_edit_row_id

                    )
                row_id = int(id_map[label])
                df_full = load_parking(prop_id)
                row = df_full[df_full["ID"] == row_id]
                if row.empty:
                    return (
                        *[gr.update() for _ in edit_boxes],     # 5
                        gr.update(),                            # bay_access_dropdown
                        gr.update(visible=False),               # edit_group
                        gr.update(visible=False),               # apply_btn
                        gr.update(visible=False),               # cancel_edit_btn
                        row_id or None,                                   # selected_edit_row_id

                    )
                row_dict = row.iloc[0].to_dict()
                values = [
                    str(row_dict[col]) if pd.notnull(row_dict[col]) else ""
                    for col in visible_columns
                ]
                edit_updates = []
                for i, col in enumerate(visible_columns):
                    if col == "BayAccessNo":
                        edit_updates.append(
                            gr.update(value=values[i], interactive=False)
                        )
                    else:
                        edit_updates.append(gr.update(value=values[i], interactive=True))

                # NEW: use APGroups (group names) for bay_access_dropdown where available
                groups = apgroups_by_prop.get(int(prop_id), []) if prop_id else []
                dropdown_choices = []
                use_groups = False

                if groups:
                    # Groups is list of (group_name, aps_string)
                    dropdown_choices = [g[0] for g in groups]
                    use_groups = True
                else:
                    # Fallback to the previous Restricted Access Points list if no APGroups defined
                    conn = get_connection()
                    q = (
                        'SELECT "AccessPointID", "NameOfAccessPoint" '
                        'FROM accesspoint '
                        'WHERE "UserID" = %s AND "PropID" = %s AND "RestrictedAP" = TRUE'
                    )
                    access_df = pd.read_sql(q, conn, params=(_uid(), prop_id))
                    conn.close()

                    if not access_df.empty:
                        id_to_label = {
                            str(r["AccessPointID"]): f'{r["NameOfAccessPoint"]} ({r["AccessPointID"]})'
                            for _, r in access_df.iterrows()
                        }
                        dropdown_choices = list(id_to_label.values())
                    else:
                        id_to_label = {}
                        dropdown_choices = []

                # Parse existing BayAccessNo to preselect:
                raw_bayaccess = str(values[bayaccess_idx] or "")

                if use_groups:
                    # Store and restore as group names, comma-separated
                    # Example stored: "Front Gates, Basement Doors"
                    raw_items = [s.strip() for s in raw_bayaccess.split(",") if s.strip()]
                    preselected = [name for name in raw_items if name in dropdown_choices]
                else:
                    # Backwards-compatible: choose labels by AccessPointID patterns
                    # Primary: match "(123456)"
                    ids_in_text = re.findall(r"\((\d+)\)", raw_bayaccess)
                    # Fallback: accept plain comma-separated digits "123,456"
                    if not ids_in_text:
                        ids_in_text = [s.strip() for s in raw_bayaccess.split(",") if s.strip().isdigit()]
                    preselected = [id_to_label[i] for i in ids_in_text if i in id_to_label]

                # Always show the dropdown (even if no available groups/APs) when editing a row
                bay_dropdown_update = gr.update(
                    choices=dropdown_choices,
                    value=preselected,
                    visible=True,
                )


                return (
                   *edit_updates,                 # 5 x edit_boxes
                   bay_dropdown_update,           # 1
                   gr.update(visible=True),       # edit_group
                   gr.update(interactive=True, visible=True),  # apply_btn
                   gr.update(visible=True),       # cancel_edit_btn
                   row_id or None,                # selected_edit_row_id
                )

            def _reactivate_delete_and_add_on_delete():
                return gr.update(interactive=True), gr.update(interactive=True)


            def _combined_tn_lease(tenant_name: str, lease_id: str) -> str:
                tn = str(tenant_name or "")
                li = str(lease_id or "")
                return f"{tn} ({li})"

            # 2. Main logic for showing edit UI + deactivate Delete selector & Add
            def _toggle_delete_and_add_on_edit(row_choice):
                if row_choice:
                    return gr.update(interactive=False), gr.update(
                        interactive=False
                    )  # delete_selector, add_btn
                return gr.update(interactive=True), gr.update(interactive=True)


            # BayAccessNo index in visible_columns
            bayaccess_idx = visible_columns.index("BayAccessNo")
            
            def _labels_to_bayaccess_text(selected_labels):
                """
                selected_labels look like: ['Main Gate (123)', 'Basement Turnstile (456)']
                Save exactly that, comma-separated, preserving selection order and de-duping.
                """
                cleaned = [lbl.strip() for lbl in (selected_labels or []) if isinstance(lbl, str) and lbl.strip()]
                seen = set()
                ordered = []
                for lbl in cleaned:
                    if lbl not in seen:
                        ordered.append(lbl)
                        seen.add(lbl)
                return ", ".join(ordered)
            


            tenantname_idx = visible_columns.index("TenantName")
            leaseid_idx = visible_columns.index("LeaseID")
            bayno_idx      = visible_columns.index("BayNo")

            # ----------- APPLY EDITS (shows menu if needed; no write yet when menu shown) -----------
            def _split_combined_tenant_lease(combined: str) -> tuple[str, str]:
                """
                Split 'TenantName (LeaseID)' into ('TenantName', 'LeaseID').
                Tolerant of extra spaces and empty parts. Returns ('', '') for bare '()' or empty input.
                """
                s = (combined or "").strip()
                if not s:
                    return "", ""
                if "(" in s and s.endswith(")"):
                    tn = s[: s.rfind("(")].strip()
                    li = s[s.rfind("(") + 1 : -1].strip()
                    return tn, li
                # If it doesn't strictly match the pattern, treat whole thing as TenantName and empty lease
                return s, ""

            # --- Choice dispatcher: routes to the correct function based on the menu selection ---
            def _apply_choice_dispatch(lease_menu_value, *rest):

                sel = (str(lease_menu_value or "").strip().lower())
                print("update running through _apply_choice_dispatch")
                print("This is the selected option : ", sel)
                args = (*rest, lease_menu_value)
                if sel.startswith("option 1") or sel.startswith("option 2"):
                    print("Running option 1 or 2")
                    return apply_edits_with_choice(*args)
                elif sel.startswith("option 3"):
                    print("Running option 3")
                    return apply_edits(*args)

            def apply_edits(*vals):
                # vals = edit_boxes... + selected_prop_id + selected_edit_row_id + choice_label
                *edit_vals, prop_id, edit_row_id, choice_label = vals
                
                print("update running through apply_edits")
                try:
                    if not prop_id or not edit_row_id:
                        raise Exception("No property or row selected.")
                    prop_id = int(prop_id)
                    edit_row_id = int(edit_row_id)

                    df = load_parking(prop_id)
                    row_before = df[df["ID"] == edit_row_id]

                    tn_before = (
                        str(row_before.iloc[0].get("TenantName") or "").strip()
                        if not row_before.empty
                        else ""
                    )
                    li_before = (
                        str(row_before.iloc[0].get("LeaseID") or "").strip()
                        if not row_before.empty
                        else ""
                    )
                    before_sync_value = f"{tn_before} ({li_before})".strip()
                    bn_before = (
                        str(row_before.iloc[0].get("BayNo") or "")
                        if not row_before.empty
                        else ""
                    )

                    print(
                        f"[DEBUG] Starting TenantName(LeaseID) FROM DB (Before Sync): {before_sync_value}"
                    )

                    # Apply edits in-memory
                    mask = df["ID"] == edit_row_id
                    for i, col in enumerate(visible_columns):
                        df.loc[mask, col] = edit_vals[i]

                    # AFTER target from inputs
                    tn_after_input = str(edit_vals[tenantname_idx] or "").strip()
                    li_after_input = str(edit_vals[leaseid_idx] or "").strip()
                    bn_after_input = str(edit_vals[bayno_idx] or "")
                    after_sync_target = _combined_tn_lease(
                        tn_after_input, li_after_input
                    ).strip()
                    print(
                        f"[DEBUG] Ending TenantName(LeaseID) BEFORE sync (After Sync): {after_sync_target}"
                    )

                    # Persist parkingbays only (no tenantuser change on this path)
                    df = sync_parking(df, prop_id)

                    # If BayNo changed, cascade to tenantuser.BayAllocated
                    try:
                        if bn_before.strip() != bn_after_input.strip():
                            _update_tenantuser_bayallocated_on_bayno_change(
                                _uid(), prop_id, edit_row_id, bn_before, bn_after_input
                            )
                    except Exception as e:
                        print(f"❌ BayAllocated cascade failed (no-menu path): {e}")

                    # AFTER-SYNC log
                    row_after = df[df["ID"] == edit_row_id]
                    if not row_after.empty:
                        tn_after = str(row_after.iloc[0].get("TenantName") or "")
                        li_after = str(row_after.iloc[0].get("LeaseID") or "")
                        after_sync_value = f"{tn_after} ({li_after})"
                        print(
                            f"[DEBUG] Ending TenantName(LeaseID) AFTER sync:  {after_sync_value}"
                        )

                    # success UI
                    row_labels = [
                        f"{row['BayNo'] or '-'} ({row['BayType'] or '-'}) [{row['ID']}]"
                        for _, row in df.iterrows()
                    ]
                    id_map = {
                        label: row["ID"]
                        for label, row in zip(row_labels, df.to_dict("records"))
                    }
                    dropdown_choices = row_labels
                    clears = [
                        gr.update(value="", interactive=True)
                        if col != "BayAccessNo"
                        else gr.update(value="", interactive=False)
                        for col in visible_columns
                    ]
                    df_to_show = df[visible_columns].rename(columns=display_labels)
                    return (
                        df_to_show,
                        gr.update(choices=dropdown_choices, visible=True, value=None),
                        gr.update(visible=True, interactive=True),
                        gr.update(
                            choices=dropdown_choices,
                            visible=True,
                            value=None,
                            interactive=True,
                        ),
                        gr.update(visible=False),  # edit_group
                        gr.update(visible=False),  # apply_btn
                        gr.update(visible=False),  # cancel_edit_btn
                        *clears,
                        gr.update(value="✅ Edit applied successfully.", visible=True),
                        id_map,
                        None,                      # selected_edit_row_id                        
                    )

                except Exception as e:
                    print("❌ Exception in apply_edits:", e)
                    return (
                        gr.update(),  # live_table
                        gr.update(),  # row_selector
                        gr.update(),  # add_btn
                        gr.update(),  # delete_selector
                        gr.update(),  # edit_group
                        gr.update(),  # apply_btn
                        gr.update(),  # cancel_edit_btn
                        *[gr.update() for _ in visible_columns],  # edit boxes
                        gr.update(value=f"❌ Error during apply edit: {e}", visible=True),
                        {},          # id_map_state
                        None,        # selected_edit_row_id
                    )


            def apply_edits_with_choice(*vals):
                # vals = edit_boxes... + selected_prop_id + selected_edit_row_id + lease_menu_choice
                *edit_vals, prop_id, edit_row_id, choice_label = vals

                print("update running through apply_edits_with_choice")
                try:
                    print(
                        f"[DEBUG] proceed.click inputs → prop_id={prop_id} ({type(prop_id)}), edit_row_id={edit_row_id} ({type(edit_row_id)}), choice={choice_label!r}"
                    )

                    # Guard: if context lost, fall back to simple apply_edits error UI
                    if prop_id in (None, "") or edit_row_id in (None, ""):
                        return apply_edits(*vals)  # choice_label is already in vals

                    prop_id = int(prop_id)
                    edit_row_id = int(edit_row_id)

                    # BEFORE snapshot
                    df = load_parking(prop_id)
                    row_before = df[df["ID"] == edit_row_id]
                    tn_before = (
                        str(row_before.iloc[0].get("TenantName") or "")
                        if not row_before.empty
                        else ""
                    )
                    li_before = (
                        str(row_before.iloc[0].get("LeaseID") or "")
                        if not row_before.empty
                        else ""
                    )
                    before_combined = f"{tn_before} ({li_before})".strip()
                    bn_before = (
                        str(row_before.iloc[0].get("BayNo") or "")
                        if not row_before.empty
                        else ""
                    )
                    bn_after_input = str(edit_vals[bayno_idx] or "")

                    # Apply edits in-memory
                    mask = df["ID"] == edit_row_id
                    for i, col in enumerate(visible_columns):
                        df.loc[mask, col] = edit_vals[i]

                    # AFTER target
                    tn_after_input = str(edit_vals[tenantname_idx] or "")
                    li_after_input = str(edit_vals[leaseid_idx] or "")
                    after_combined = f"{tn_after_input} ({li_after_input})".strip()
                    print("BEFORE APPLY_CHOICE_P&TENUSER: before edit: ", before_combined, "  After edit", after_combined, "option (ignored): ", choice_label)
                    
                    # Transaction with global behaviour (Option 1 semantics)
                    df = _apply_choice_parking_and_tenantuser(
                        df, prop_id, before_combined, after_combined, choice_label
                    )

                    # Cascade BayAllocated as well if BayNo changed
                    try:
                        if bn_before.strip() != bn_after_input.strip():
                            _update_tenantuser_bayallocated_on_bayno_change(
                                _uid(), prop_id, edit_row_id, bn_before, bn_after_input
                            )
                    except Exception as e:
                        print(f"❌ BayAllocated cascade failed (menu path): {e}")

                    # Success UI
                    row_labels = [
                        f"{row['BayNo'] or '-'} ({row['BayType'] or '-'}) [{row['ID']}]"
                        for _, row in df.iterrows()
                    ]
                    id_map = {
                        label: row["ID"]
                        for label, row in zip(row_labels, df.to_dict("records"))
                    }
                    dropdown_choices = row_labels
                    clears = [
                        gr.update(value="", interactive=True)
                        if col != "BayAccessNo"
                        else gr.update(value="", interactive=False)
                        for col in visible_columns
                    ]
                    df_to_show = df[visible_columns].rename(columns=display_labels)
                    return (
                        df_to_show,
                        gr.update(choices=dropdown_choices, visible=True, value=None),
                        gr.update(visible=True, interactive=True),
                        gr.update(
                            choices=dropdown_choices,
                            visible=True,
                            value=None,
                            interactive=True,
                        ),
                        gr.update(visible=False),  # edit_group
                        gr.update(visible=False),  # apply_btn
                        gr.update(visible=False),  # cancel_edit_btn
                        *clears,
                        gr.update(value="✅ Edit applied successfully.", visible=True),
                        id_map,
                        None,                      # selected_edit_row_id
                    )
                except Exception as e:
                    print("❌ Exception in apply_edits_with_choice:", e)
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        *[gr.update() for _ in visible_columns],
                        gr.update(value=f"❌ Error during apply edit with choices: {e}", visible=True),
                        {},
                        None,
                    )


            # ----------- ADD ROW -----------
            def add_row(prop_id, prop_name):
                new_id = generate_id()
                uid = _uid()
                new_row = {
                    "ID": new_id,
                    "PropID": prop_id,
                    "Name": prop_name,
                    "UserID": uid,
                    "BayType": None,
                    "BayNo": None,
                    "TenantName": "",
                    "LeaseEnd": None,
                    "LeaseID": None,
                    "BayAccessNo": "",
                }
                df = load_parking(prop_id)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df = sync_parking(df, prop_id)
                row_labels = [
                    f"{row['BayNo'] or '-'} ({row['BayType'] or '-'}) [{row['ID']}]"
                    for _, row in df.iterrows()
                ]
                id_map = {
                    label: row["ID"]
                    for label, row in zip(row_labels, df.to_dict("records"))
                }
                dropdown_choices = row_labels
                df_to_show = df[visible_columns].rename(columns=display_labels)
                return (
                    df_to_show,
                    gr.update(
                        choices=dropdown_choices,
                        visible=True,
                        value=None,
                        interactive=not df.empty,
                    ),  # row_selector refreshed
                    gr.update(visible=True, interactive=True),  # add_btn active
                    gr.update(
                        choices=dropdown_choices,
                        visible=True,
                        value=None,
                        interactive=not df.empty,
                    ),  # delete_selector refreshed
                    id_map,
                    gr.update(value="✅ Row added.", visible=False),
                )


            # ----------- CANCEL EDIT FUNCTION & BUTTON -----------
            def on_cancel_edit(live_table, selected_prop_id, selected_prop_name, id_map_state):
                if selected_prop_id is None:
                    return (
                        gr.update(visible=False),  # edit_group
                        gr.update(visible=False),  # apply_btn
                        gr.update(visible=False),  # cancel_edit_btn
                        gr.update(value="", visible=False),  # edit_feedback
                        gr.update(choices=[], visible=False, value=None),  # row_selector
                        gr.update(visible=False, interactive=False),  # add_btn
                        gr.update(choices=[], visible=False, value=None),  # delete_selector
                        *[
                            gr.update(value="", interactive=True)
                            if col != "BayAccessNo"
                            else gr.update(value="", interactive=False)
                            for col in visible_columns
                        ],
                        None,
                    )
                df = load_parking(selected_prop_id)
                row_labels = [
                    f"{row['BayNo'] or '-'} ({row['BayType'] or '-'}) [{row['ID']}]"
                    for _, row in df.iterrows()
                ]
                id_map = {
                    label: row["ID"]
                    for label, row in zip(row_labels, df.to_dict("records"))
                }
                dropdown_choices = row_labels
                df_to_show = df[visible_columns].rename(columns=display_labels)
                clears = [
                    gr.update(value="", interactive=True)
                    if col != "BayAccessNo"
                    else gr.update(value="", interactive=False)
                    for col in visible_columns
                ]
                return (
                    gr.update(visible=False),  # edit_group hidden
                    gr.update(visible=False),  # apply_btn hidden
                    gr.update(visible=False),  # cancel_edit_btn hidden
                    gr.update(value="", visible=False),  # edit_feedback hidden
                    gr.update(choices=dropdown_choices, visible=True, value=None),  # row_selector
                    gr.update(visible=True, interactive=True),  # add_btn
                    gr.update(
                        choices=dropdown_choices, visible=True, value=None, interactive=True
                    ),  # delete_selector
                    *clears,
                    None,
                )

            # ----------- DELETE ROW -----------
            def delete_row(label, prop_id, id_map):
                if not label or label not in id_map:
                    df = load_parking(prop_id)
                    row_labels = [
                        f"{row['BayNo'] or '-'} ({row['BayType'] or '-'}) [{row['ID']}]"
                        for _, row in df.iterrows()
                    ]
                    id_map_new = {
                        l: row["ID"] for l, row in zip(row_labels, df.to_dict("records"))
                    }
                    dropdown_choices = row_labels
                    df_to_show = df[visible_columns].rename(columns=display_labels)
                    return (
                        df_to_show,
                        gr.update(
                            choices=dropdown_choices,
                            visible=True,
                            value=None,
                            interactive=not df.empty,
                        ),
                        gr.update(visible=True, interactive=True),
                        gr.update(
                            choices=dropdown_choices,
                            visible=True,
                            value=None,
                            interactive=not df.empty,
                        ),
                        id_map_new,
                        gr.update(value="", visible=False),
                    )
                row_id = int(id_map[label])
                df = load_parking(prop_id)
                df = df[df["ID"] != row_id]
                df = sync_parking(df, prop_id)
                df = load_parking(prop_id)
                row_labels = [
                    f"{row['BayNo'] or '-'} ({row['BayType'] or '-'}) [{row['ID']}]"
                    for _, row in df.iterrows()
                ]
                id_map = {
                    label: row["ID"]
                    for label, row in zip(row_labels, df.to_dict("records"))
                }
                dropdown_choices = row_labels
                df_to_show = df[visible_columns].rename(columns=display_labels)
                return (
                    df_to_show,
                    gr.update(
                        choices=dropdown_choices,
                        visible=True,
                        value=None,
                        interactive=not df.empty,
                    ),  # row_selector refreshed
                    gr.update(visible=True, interactive=True),  # add_btn active
                    gr.update(
                        choices=dropdown_choices,
                        visible=True,
                        value=None,
                        interactive=not df.empty,
                    ),  # delete_selector refreshed
                    id_map,
                    gr.update(value="✅ Row deleted.", visible=True),
                )



            def _freeze_edit_and_add_on_delete(label):
                if label:
                    return gr.update(interactive=False), gr.update(
                        interactive=False
                    )  # row_selector, add_btn
                return gr.update(), gr.update()

            def _unfreeze_edit_and_add_after_delete():
                return gr.update(interactive=True), gr.update(
                    interactive=True
                )  # row_selector, add_btn
            
            
            def _prime_uid_then_fill_properties():

                # give callback/env time to become available (HF cold start can be slow)
                deadline = time.time() + 5.0
                while resolve_and_cache_uid(get_user_id) is None and time.time() < deadline:
                    time.sleep(0.1)
            
                uid = _uid()
            
                # If still no UID, keep the dropdown empty & disabled (no 'options' reference!)
                if uid is None:
                    return gr.update(choices=[], value=None, interactive=False)
            
                # Now UID is present → fetch and populate
                df = load_property_list()
                options = [f"{row['Name']} ({row['Owner']})" for _, row in df.iterrows()]
                # refresh the shared map in-place
                new_map = {opt: (row["PropID"], row["Name"]) for opt, (_, row) in zip(options, df.iterrows())}
                prop_map.clear()
                prop_map.update(new_map)
                apgroups_by_prop.clear()   # NEW: reset APGroups cache on reload


                return gr.update(choices=options, value=None, interactive=bool(options))            

            def _refresh_after_sort(order_choice, prop_id, prop_name):
                if not prop_id:
                    # nothing selected yet → keep UI as-is
                    return (
                        gr.update(),  # live_table
                        prop_id,
                        prop_name,
                        gr.update(),  # row_selector
                        gr.update(),  # add_btn
                        gr.update(),  # delete_selector
                        gr.update(),  # id_map_state
                        gr.update(value="", visible=False),  # edit_feedback
                        gr.update(value=order_choice or "Bay No", visible=True),  # sort_by
                    )
            
                df = load_parking(prop_id)
                df = _apply_ordering(df, order_choice or "Bay No")
            
                row_labels = [
                    f"{row['BayNo'] or '-'} ({row['BayType'] or '-'})"
                    for _, row in df.iterrows()
                ]
                id_map = {
                    lbl: row["ID"]
                    for lbl, row in zip(row_labels, df.to_dict("records"))
                }
                df_to_show = df[visible_columns].rename(columns=display_labels)
            
                return (
                    gr.update(value=df_to_show, visible=True),  # live_table
                    prop_id,                                    # selected_prop_id (State)
                    prop_name,                                  # selected_prop_name (State)
                    gr.update(                                  # row_selector
                        choices=row_labels,
                        visible=True,
                        value=None,
                        interactive=not df.empty,
                    ),
                    gr.update(visible=True, interactive=True),  # add_btn
                    gr.update(                                  # delete_selector
                        choices=row_labels,
                        visible=True,
                        value=None,
                        interactive=not df.empty,
                    ),
                    id_map,                                     # id_map_state
                    gr.update(value="", visible=False),         # edit_feedback
                    gr.update(value=order_choice or "Bay No", visible=True),  # sort_by
                )


            # ========== APP ==========
 
            prop_map = {}
            apgroups_by_prop = {}  
            print("DEBUG (prime): starting with UID =", _uid())
            prop_selector = gr.Dropdown(label="Select Property (Owner)", choices=[], value=None)            

            btn_reload_props = gr.Button("Reload Properties", variant="secondary")
            btn_reload_props.click(_prime_uid_then_fill_properties, outputs=[prop_selector])

            sort_by = gr.Radio(
                label="Order table by",
                choices=["Bay No", "Tenant Name", "Unit ID"],
                value="Bay No",
                visible=False 
            )

            live_table = gr.Dataframe(label="List of All Parking Bays at Property", visible=False, interactive=False)
            demo.load(_prime_uid_then_fill_properties, inputs=[], outputs=[prop_selector])
            
            add_btn = gr.Button("Add New Bay", interactive=False, visible=False)
            with gr.Row():
                row_selector   = gr.Dropdown(label="Select Bay to edit",    choices=[], value=None, interactive=False, visible=False)
                delete_selector= gr.Dropdown(label="Select Bay to delete",  choices=[], value=None, interactive=False, visible=False)

            edit_feedback = gr.Textbox(label="Edit Status", visible=False)
            selected_prop_id   = gr.State(value=None)
            selected_prop_name = gr.State(value=None)
            id_map_state       = gr.State({})
            selected_edit_row_id = gr.State(value=None)
#            pending_edit_row_id  = gr.State(value=None)

            with gr.Group(visible=False) as edit_group:
                gr.Markdown("""<div style="background-color:#666666; color:#FFFFFF; padding:10px; border-radius:0px;">Details of Parking Bay</div>""")
                with gr.Row():
                    box_bayno = gr.Textbox(label=display_labels["BayNo"])
                    box_baytype = gr.Dropdown(
                        label=display_labels["BayType"],
                        choices=[
                            "Visitors Bay",
                            "Open Bay",
                            "Covered Bay",
                            "Basement Bay",
                            "Garage Bay",
                            "Loading Bay",
                            "Other",
                        ],
                        allow_custom_value=True,
                    )
                    
                    bay_access_dropdown = gr.Dropdown(
                        label="Select AP-Groups for Bay",
                        choices=[""],
                        value="",
                        multiselect=True,
                        visible=False,
                    )
                    
                    box_bayaccessno = gr.Textbox(label=display_labels["BayAccessNo"], interactive=False)                    
                gr.Markdown(
                    """<div style="background-color:#666666; color:#FFFFFF; padding:10px; border-radius:0px;">Details of Lease / Person Connected to Bay</div>"""
                )
                with gr.Row():
                    box_leaseid = gr.Textbox(label="Name of Unit / LeaseID")
                    box_tenantname = gr.Textbox(label="Tenant / Person's Name")

                box_map = {
                    "BayType": box_baytype,
                    "BayNo": box_bayno,
                    "BayAccessNo": box_bayaccessno,
                    "LeaseID": box_leaseid,
                    "TenantName": box_tenantname,
                }
                edit_boxes = [box_map[col] for col in visible_columns]

#                debug_flag = gr.Markdown(value="", visible=True)

                with gr.Row():
                    apply_btn = gr.Button("Apply Edits", interactive=False, visible=True)
                    cancel_edit_btn = gr.Button("Cancel Edit")



            # BINDERS 


            row_selector.change(
                lambda *args: gr.update(value="", visible=False),
                inputs=[row_selector],
                outputs=[edit_feedback],
                queue=False,
                api_name=False,
            )

            row_selector.change(
                on_row_selected,
                inputs=[row_selector, live_table, selected_prop_id, id_map_state],
                outputs=edit_boxes                                   # 5 items
                + [bay_access_dropdown, edit_group, apply_btn, cancel_edit_btn, selected_edit_row_id],
            ).then(
                _toggle_delete_and_add_on_edit,
                inputs=[row_selector],
                outputs=[delete_selector, add_btn],
            )


            bay_access_dropdown.change(
                _labels_to_bayaccess_text,
                inputs=bay_access_dropdown,
                outputs=edit_boxes[bayaccess_idx],
            )


            # AFTER: route through the helper and EXCLUDE lease_menu_choice
            apply_btn.click(
                _maybe_show_lease_menu,
                inputs=edit_boxes + [selected_prop_id, selected_edit_row_id],
                outputs=[
                    live_table, row_selector, add_btn, delete_selector,
                    edit_group, apply_btn, cancel_edit_btn,
                    *edit_boxes, edit_feedback, id_map_state, selected_edit_row_id,
                ],
            )


            prop_selector.change(
                on_property_selected,
                inputs=[prop_selector, sort_by],
                outputs=[
                    live_table,
                    selected_prop_id,
                    selected_prop_name,
                    row_selector,
                    add_btn,
                    delete_selector,
                    id_map_state,
                    edit_feedback,
                    sort_by,
                ],
            )

            cancel_edit_btn.click(
                on_cancel_edit,
                inputs=[live_table, selected_prop_id, selected_prop_name, id_map_state],
                outputs=[
                    edit_group,
                    apply_btn,
                    cancel_edit_btn,
                    edit_feedback,
                    row_selector,
                    add_btn,
                    delete_selector,
                    *edit_boxes,
                    selected_edit_row_id,
                ],
            ).then(_reactivate_delete_and_add_on_delete, inputs=[], outputs=[delete_selector, add_btn])


            add_btn.click(
                add_row,
                inputs=[selected_prop_id, selected_prop_name],
                outputs=[
                    live_table,
                    row_selector,
                    add_btn,
                    delete_selector,
                    id_map_state,
                    edit_feedback,
                ],
            )

            sort_by.change(
                _refresh_after_sort,
                inputs=[sort_by, selected_prop_id, selected_prop_name],
                outputs=[  # same shape as on_property_selected
                    live_table,
                    selected_prop_id,
                    selected_prop_name,
                    row_selector,
                    add_btn,
                    delete_selector,
                    id_map_state,
                    edit_feedback,
                    sort_by,
                ],
            )

            def _delete_row_guard(label, prop_id, id_map):
                # Avoid extra DB work on programmatic updates (value=None after property select)
                if not label or label not in id_map:
                    return (
                        gr.update(),  # keep live_table as-is
                        gr.update(),  # keep row_selector as-is
                        gr.update(),  # keep add_btn as-is
                        gr.update(),  # keep delete_selector as-is
                        gr.update(),  # keep id_map_state as-is
                        gr.update(value="", visible=False),  # clear feedback
                    )
                # Valid selection -> do the actual deletion
                return delete_row(label, prop_id, id_map)

            delete_selector.change(
                _freeze_edit_and_add_on_delete, inputs=[delete_selector], outputs=[row_selector, add_btn]
            ).then(
                _delete_row_guard,
                inputs=[delete_selector, selected_prop_id, id_map_state],
                outputs=[live_table, row_selector, add_btn, delete_selector, id_map_state, edit_feedback],
            ).then(
                _unfreeze_edit_and_add_after_delete, inputs=[], outputs=[row_selector, add_btn]
            )

    return demo

