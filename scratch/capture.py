import os
import time
import subprocess
import sys
import urllib.request

# Ensure streamlit is installed
try:
    import streamlit
except ImportError:
    print("Installing Streamlit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])

# Ensure playwright is installed
try:
    import playwright
except ImportError:
    print("Installing Playwright...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])

from playwright.sync_api import sync_playwright

def capture_screenshots():
    # Start local streamlit server
    print("Starting local Streamlit server...")
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true"]
    process = subprocess.Popen(cmd)
    
    # Wait for the local server to respond
    url = "http://localhost:8501"
    print(f"Waiting for local server to respond at {url}...")
    for _ in range(30):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print("Server is responding!")
                    break
        except Exception:
            pass
        time.sleep(1)
    
    # Absolute paths for uploads
    defocus_path = os.path.abspath("input/eiffel_defocus.jpg")
    mixed_path = os.path.abspath("input/eiffel_mixed.jpg")
    
    os.makedirs("comparisons", exist_ok=True)
    
    try:
        with sync_playwright() as p:
            print("Launching browser...")
            browser = p.chromium.launch(headless=True)
            # Use a tall viewport so the entire Eiffel Tower side-by-side fits naturally
            context = browser.new_context(viewport={"width": 1400, "height": 1800})
            page = context.new_page()
            
            print(f"Navigating to {url}...")
            page.goto(url)
            
            # Wait for file uploader to be attached
            print("Waiting for file uploader...")
            page.wait_for_selector('input[type="file"]', state="attached", timeout=30000)
            
            # 1. Defocus Image
            print("Uploading eiffel_defocus.jpg...")
            file_input = page.locator('input[type="file"]')
            file_input.set_input_files(defocus_path)
            
            # Click "Start Restoration" button
            print("Clicking Start Restoration button...")
            page.wait_for_selector('button:has-text("Start Restoration")', timeout=20000)
            page.click('button:has-text("Start Restoration")')
            
            # Wait for processing to complete
            print("Waiting for restoration to complete...")
            page.wait_for_selector('text="Original Blurry Image"', timeout=60000)
            time.sleep(3)
            
            # Hide the header and footer for clean screenshot (NO scale transforms!)
            page.evaluate("""() => {
                const header = document.querySelector('header');
                if (header) header.style.display = 'none';
                const footer = document.querySelector('footer');
                if (footer) footer.style.display = 'none';
                
                // Remove top padding of main block to pull content up
                const mainContent = document.querySelector('.block-container');
                if (mainContent) {
                    mainContent.style.paddingTop = '1rem';
                    mainContent.style.paddingBottom = '1rem';
                }
            }""")
            time.sleep(1)
            
            # Scroll to make sure the side-by-side columns are centered in the screenshot
            page.locator('text="Processing File:"').scroll_into_view_if_needed()
            time.sleep(1)
            
            print("Capturing Defocus screenshot...")
            page.screenshot(path="comparisons/screenshot_eiffel_defocus.png", full_page=False)
            print("Saved comparisons/screenshot_eiffel_defocus.png")
            
            # 2. Mixed Image
            print("Reloading page and uploading eiffel_mixed.jpg...")
            page.reload()
            page.wait_for_selector('input[type="file"]', state="attached", timeout=20000)
            
            file_input = page.locator('input[type="file"]')
            file_input.set_input_files(mixed_path)
            
            print("Clicking Start Restoration button...")
            page.wait_for_selector('button:has-text("Start Restoration")', timeout=20000)
            page.click('button:has-text("Start Restoration")')
            
            print("Waiting for mixed restoration...")
            page.wait_for_selector('text="Original Blurry Image"', timeout=60000)
            time.sleep(3)
            
            page.evaluate("""() => {
                const header = document.querySelector('header');
                if (header) header.style.display = 'none';
                const footer = document.querySelector('footer');
                if (footer) footer.style.display = 'none';
                
                const mainContent = document.querySelector('.block-container');
                if (mainContent) {
                    mainContent.style.paddingTop = '1rem';
                    mainContent.style.paddingBottom = '1rem';
                }
            }""")
            time.sleep(1)
            
            page.locator('text="Processing File:"').scroll_into_view_if_needed()
            time.sleep(1)
            
            print("Capturing Mixed screenshot...")
            page.screenshot(path="comparisons/screenshot_eiffel_mixed.png", full_page=False)
            print("Saved comparisons/screenshot_eiffel_mixed.png")
            
            browser.close()
    finally:
        print("Terminating local Streamlit server...")
        process.terminate()
        process.wait()
        print("Finished capturing all screenshots successfully!")

if __name__ == "__main__":
    capture_screenshots()
