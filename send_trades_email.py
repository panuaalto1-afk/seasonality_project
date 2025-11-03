#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
send_trades_email.py - Lähetä trade candidates emaililla

Yksinkertainen skripti, joka lähettää viimeisimmän trade_candidates.csv
emaililla liitteenä.

Käyttö:
    python send_trades_email.py
"""

import os
import glob
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

# ==================== ASETUKSET ====================
# MUUTA NÄMÄ OMIIN TIETOIHISI:

SENDER_EMAIL = "panu.aalto1@gmail.com"           # ⬅️ MUUTA: lähettäjän Gmail
SENDER_PASSWORD = "yybj gfjb lxbx yytf"         # ⬅️ MUUTA: Gmail App Password (16 merkkiä)
RECEIVER_EMAIL = "panu.aalto1@gmail.com"         # ⬅️ MUUTA: puhelimesi email

# Gmail SMTP asetukset (nämä voi jättää)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ===================================================

def find_latest_action_dir():
    """Etsi viimeisin actions-kansio"""
    pattern = "seasonality_reports/runs/*/actions/*"
    dirs = glob.glob(pattern)
    
    if not dirs:
        print("[ERROR] Ei löytynyt actions-kansioita!")
        return None
    
    # Järjestä muokkausajan mukaan, viimeisin ensin
    latest = max(dirs, key=os.path.getmtime)
    return latest

def create_email_message(action_dir):
    """Luo email-viesti liitteineen"""
    
    # Tiedostot jotka lähetetään
    files_to_attach = [
        "trade_candidates.csv",
        "action_plan.txt",
        "sell_candidates.csv"
    ]
    
    # Luo viesti
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = f"📈 Trade Candidates - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Laske yhteenveto action_plan.txt:stä
    action_plan_path = os.path.join(action_dir, "action_plan.txt")
    summary = ""
    
    if os.path.exists(action_plan_path):
        try:
            with open(action_plan_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Etsi yhteenveto (viimeiset rivit)
                if "YHTEENVETO" in content or "SUMMARY" in content:
                    lines = content.split('\n')
                    # Ota viimeiset 15 riviä
                    summary = "\n".join(lines[-15:])
        except:
            pass
    
    # Viestin body
    body = f"""
Seasonality Trading Signals
Generated: {datetime.fromtimestamp(os.path.getmtime(action_dir)).strftime('%Y-%m-%d %H:%M:%S')}

{summary if summary else 'See attachments for details.'}

---
Automated by seasonality_project
"""
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Liitä tiedostot
    attached_count = 0
    for filename in files_to_attach:
        file_path = os.path.join(action_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"[SKIP] {filename} - ei löydy")
            continue
        
        try:
            with open(file_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={filename}')
            msg.attach(part)
            
            file_size = os.path.getsize(file_path)
            print(f"[OK] Liitetty: {filename} ({file_size} bytes)")
            attached_count += 1
            
        except Exception as e:
            print(f"[ERROR] {filename} liittäminen epäonnistui: {e}")
    
    if attached_count == 0:
        print("[ERROR] Ei liitettyjä tiedostoja!")
        return None
    
    return msg

def send_email(msg):
    """Lähetä email Gmail SMTP:n kautta"""
    
    try:
        print(f"\n[INFO] Yhdistetään: {SMTP_SERVER}:{SMTP_PORT}")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        
        print(f"[INFO] Kirjaudutaan: {SENDER_EMAIL}")
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        print(f"[INFO] Lähetetään: {SENDER_EMAIL} → {RECEIVER_EMAIL}")
        server.send_message(msg)
        server.quit()
        
        print(f"[OK] ✅ Email lähetetty onnistuneesti!")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("\n[ERROR] ❌ Kirjautuminen epäonnistui!")
        print("Tarkista:")
        print("  1. SENDER_EMAIL on oikein")
        print("  2. SENDER_PASSWORD on Gmail App Password (ei tavallinen salasana)")
        print("  3. 2-Factor Authentication on päällä Gmailissa")
        print("\nLuo App Password: https://myaccount.google.com/apppasswords")
        return False
        
    except smtplib.SMTPException as e:
        print(f"\n[ERROR] ❌ SMTP virhe: {e}")
        return False
        
    except Exception as e:
        print(f"\n[ERROR] ❌ Emailin lähetys epäonnistui: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("📧 TRADE CANDIDATES EMAIL SENDER")
    print("="*70 + "\n")
    
    # Tarkista asetukset
    if SENDER_EMAIL == "your.email@gmail.com":
        print("❌ VIRHE: Muokkaa SENDER_EMAIL skriptissä!")
        print("Avaa: notepad send_trades_email.py")
        print("Muuta rivit 18-20\n")
        return
    
    # Etsi viimeisin actions-kansio
    print("[STEP 1/3] Etsitään viimeisintä trade_candidates.csv...")
    action_dir = find_latest_action_dir()
    
    if not action_dir:
        print("\n❌ Ei löytynyt actions-kansioita.\n")
        return
    
    print(f"[OK] Löytyi: {action_dir}")
    mod_time = datetime.fromtimestamp(os.path.getmtime(action_dir))
    print(f"     Muokattu: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Luo email-viesti
    print(f"\n[STEP 2/3] Luodaan email-viesti liitteineen...")
    msg = create_email_message(action_dir)
    
    if not msg:
        print("\n❌ Emailin luominen epäonnistui.\n")
        return
    
    # Lähetä email
    print(f"\n[STEP 3/3] Lähetetään email...")
    success = send_email(msg)
    
    if success:
        print("\n" + "="*70)
        print("✅ VALMIS!")
        print("="*70)
        print("\nTarkista puhelimesi email muutaman minuutin kuluttua.")
        print("Liitteet:")
        print("  - trade_candidates.csv")
        print("  - action_plan.txt")
        print("  - sell_candidates.csv")
        print("\n" + "="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("❌ LÄHETYS EPÄONNISTUI")
        print("="*70)
        print("\nTarkista asetukset ja yritä uudelleen.\n")

if __name__ == "__main__":
    main()