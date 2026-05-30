#!/usr/bin/env python3
"""Download White-Box Cartoonization weights from Google Drive with captcha bypass"""
import requests, re, os, sys

def download_file(file_id, output_path):
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # First request
    r = session.get(url, allow_redirects=True)
    
    # Check if Google shows warning/confirmation page
    if 'confirm=' in r.text or 'uc-download-link' in r.text:
        token = re.search(r'confirm=([a-zA-Z0-9_-]+)', r.text)
        if token:
            confirm_url = f"https://drive.google.com/uc?export=download&confirm={token.group(1)}&id={file_id}"
            r = session.get(confirm_url, stream=True)
    
    # Download with progress
    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = min(100, downloaded * 100 // total)
                    print(f"\r⬇️ {percent}%...", end='', flush=True)
    print(f"\n✅ Downloaded: {output_path} ({downloaded/1024/1024:.1f} MB)")
    return downloaded

def main():
    FILE_ID = os.environ.get("GDRIVE_FILE_ID", "1yXSY8PZf6u42tXMsHLXow-9MLhaX-H3q")
    OUTPUT = os.environ.get("GDRIVE_OUTPUT", "wbc_models.zip")
    
    print(f"🔗 Downloading from Google Drive (ID: {FILE_ID})...")
    
    size = download_file(FILE_ID, OUTPUT)
    
    # Validate
    if size < 10 * 1024 * 1024:
        # Check if it's HTML error page
        with open(OUTPUT, 'r', errors='ignore') as f:
            content = f.read(500)
        if '<!DOCTYPE html>' in content or 'Google Drive - Quota exceeded' in content:
            print(f"❌ Downloaded HTML error page. Check Drive permissions (must be 'Anyone with link').")
            sys.exit(1)
        print(f"⚠️ File small ({size/1024/1024:.1f}MB), but continuing...")
    
    print("✅ Download validation passed")

if __name__ == "__main__":
    main()
