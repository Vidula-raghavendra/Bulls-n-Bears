import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

def indefinite_debug_scraper(query):
    """
    This function runs the scraper with a VISIBLE browser window
    and KEEPS IT OPEN until you press Enter in the terminal.
    """
    search_query = f'"{query}" stock'
    url = f"https://news.google.com/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"

    # --- Selenium Setup ---
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    service = ChromeService(ChromeDriverManager().install())
    
    driver = None
    print("--- Starting Visual Debug ---")
    
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print(f"1. Navigating to: {url}")
        driver.get(url)
        
        # --- KEY CHANGE IS HERE ---
        print("\n" + "="*50)
        print("BROWSER WINDOW IS NOW OPEN AND WILL STAY OPEN.")
        print("Take your time. You can now do the following:")
        print("  1. Go to the Chrome window that just opened.")
        print("  2. Right-click a headline and choose 'Inspect'.")
        print("  3. Find the class names for the container and headline tags.")
        print("\n===> WHEN YOU ARE FINISHED, COME BACK TO THIS TERMINAL AND PRESS 'ENTER' TO CLOSE THE BROWSER. <===")
        print("="*50 + "\n")
        
        input() # This line will pause the script indefinitely until you press Enter.
        
        # The script will only continue from here after you press Enter
        print("\n--- Resuming script to perform scrape ---")
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        headlines = []
        # Find all article containers
        article_containers = soup.find_all('div', class_='fKcn1')
        if not article_containers:
            article_containers = soup.find_all('div', class_='xT965d')

        for container in article_containers:
            headline_tag = container.find('a', class_='JtKRv')
            if headline_tag: headlines.append(headline_tag.text)
        
        print(f"Scraper found {len(headlines)} headlines from the page.")
        return headlines

    except Exception as e:
        print(f"AN ERROR OCCURRED: {e}")
        return []
    finally:
        if driver:
            print("Closing the browser now.")
            driver.quit()
        print("--- Debug Finished ---")

# --- Main part of the script ---
if __name__ == "__main__":
    test_stock = "GOOG" 
    print(f"--- Running an indefinite test for: {test_stock} ---")
    indefinite_debug_scraper(test_stock)
    print("\nCheck the terminal for scrape results.")