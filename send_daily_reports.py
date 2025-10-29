#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
send_daily_reports.py
Lähettää päivän action_plan + logs (+ kaikki .txt/.csv) sähköpostitse.

ENV muuttujat (pakolliset):
  SMTP_HOST   = "smtp.office365.com" (Outlook/Hotmail) TAI "smtp.gmail.com" (Gmail)
  SMTP_PORT   = "587" (STARTTLS)    TAI "465" (SSL)
  SMTP_USER   = lähettäjän sähköposti (esim. sama kuin vastaanottaja)
  SMTP_PASS   = salasana TAI app password

Valinnaiset:
  EMAIL_TO    = vastaanottaja, esim. "panu.aalto@windowslive.com"
  EMAIL_SUBJ_PREFIX = oletus "[Seasonality]"
"""

import os
import sys
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

def _attach_file(msg: MIMEMultipart, path: str):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        filename = os.path.basename(path)
        part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
        msg.attach(part)
        return True
    except Exception:
        return False

def _discover_action_files(actions_dir: str):
    # Yritä nimet sekä .txt että ilman päätettä
    cand = []
    for base in ("action_plan", "logs", "trade_candidates", "portfolio_after_sim"):
        for ext in ("", ".txt", ".csv"):
            p = os.path.join(actions_dir, base + ext)
            if os.path.isfile(p):
                cand.append(p)
    # lisäksi liitä kaikki .txt/.csv varmuuden vuoksi
    for name in os.listdir(actions_dir):
        if name.lower().endswith((".txt", ".csv")):
            p = os.path.join(actions_dir, name)
            if p not in cand:
                cand.append(p)
    # Poista duplikaatit säilyttäen järjestys
    seen = set(); out = []
    for p in cand:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: send_daily_reports.py <ACTIONS_DIR> [SUBJECT_TAG]")
        sys.exit(2)

    actions_dir = sys.argv[1]
    subject_tag = sys.argv[2] if len(sys.argv) >= 3 else ""

    if not os.path.isdir(actions_dir):
        print(f"[WARN] Actions dir not found: {actions_dir}")
        sys.exit(0)

    host = os.environ.get("SMTP_HOST", "")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    pwd  = os.environ.get("SMTP_PASS", "")
    to   = os.environ.get("EMAIL_TO", "panu.aalto@windowslive.com")
    subj_prefix = os.environ.get("EMAIL_SUBJ_PREFIX", "[Seasonality]")

    if not (host and port and user and pwd and to):
        print("[WARN] Missing SMTP env (SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS or EMAIL_TO). Email skipped.")
        sys.exit(0)

    date_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"{subj_prefix} Action Plan {date_str}"
    if subject_tag:
        subject += f" | {subject_tag}"

    # Rakenna viesti
    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = to
    msg["Date"] = formatdate(localtime=True)
    msg["Subject"] = subject

    files = _discover_action_files(actions_dir)
    body_lines = [
        f"Daily one-click run completed {date_str}.",
        f"Attached files from: {actions_dir}",
        "",
        "Attachments:"
    ] + [f" - {os.path.basename(p)}" for p in files]
    msg.attach(MIMEText("\n".join(body_lines), "plain", "utf-8"))

    # Liitteet
    attached = 0
    for p in files:
        if _attach_file(msg, p):
            attached += 1

    # Lähetys (SSL vai STARTTLS)
    use_ssl = (port == 465)
    if use_ssl:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx) as s:
            s.login(user, pwd)
            s.sendmail(user, [to], msg.as_string())
    else:
        with smtplib.SMTP(host, port) as s:
            s.ehlo()
            s.starttls(context=ssl.create_default_context())
            s.ehlo()
            s.login(user, pwd)
            s.sendmail(user, [to], msg.as_string())

    print(f"[OK] Email sent to {to} with {attached} attachment(s).")

if __name__ == "__main__":
    main()
