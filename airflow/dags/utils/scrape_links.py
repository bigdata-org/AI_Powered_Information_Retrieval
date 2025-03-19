from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
from aws import s3
import os
import re

def scrape_pdf_links():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)

    bucket_name = os.getenv('BUCKET_NAME')
    s3_client = s3.get_s3_client()
    try:
        # Navigate to the page
        driver.get("https://investor.nvidia.com/financial-info/quarterly-results/default.aspx")
        
        pdf_links = {}
        years_to_scrape = {"2025","2024","2023","2022","2021"}
        
        for year in years_to_scrape:
            # Wait for the dropdown to be present
            wait = WebDriverWait(driver, 10)
            dropdown_element = wait.until(EC.presence_of_element_located((By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear")))
            
            # Create Select object
            select = Select(dropdown_element)
            
            # Select by visible text
            select.select_by_visible_text(year)
            
            # Wait for page to update after selection
            time.sleep(3)
            
            # Find all accordion toggle buttons
            accordion_buttons = driver.find_elements(By.XPATH, "//button[contains(@class, 'evergreen-financial-accordion-toggle')]")
            
            year_links = {}
            
            # Expand each accordion section
            for i, button in enumerate(accordion_buttons):
                try:
                    # Get quarter name before clicking
                    quarter_name = button.find_element(By.XPATH, ".//span[@class='evergreen-accordion-title']").text

                    time.sleep(1)
                    # Try multiple click methods for better reliability
                    try:
                        # First try JavaScript click
                        driver.execute_script("arguments[0].click();", button)
                    except:
                        try:
                            # Then try regular click
                            button.click()
                        except:
                            # Finally try action chains
                            from selenium.webdriver.common.action_chains import ActionChains
                            actions = ActionChains(driver)
                            actions.move_to_element(button).click().perform()
                    
                    # Wait for content to load - longer wait for first accordion
                    wait_time = 3 if i == 0 else 1
                    time.sleep(wait_time)
                    
                    # Check if panel is expanded
                    is_expanded = button.get_attribute("aria-expanded") == "true"
                    if not is_expanded:
                        print(f"Warning: {quarter_name} panel did not expand, trying again")
                        driver.execute_script("arguments[0].click();", button)
                        time.sleep(2)
                    
                    # Find the expanded content container
                    panel_id = button.get_attribute("aria-controls")
                    panel = wait.until(EC.presence_of_element_located((By.ID, panel_id)))
                    
                    # Find all links in the expanded section
                    links = panel.find_elements(By.XPATH, ".//a[contains(@class, 'evergreen-financial-accordion-link')]")
                    
                    for link in links:
                        href = link.get_attribute('href')
                        text = link.text.strip()
                        
                        # Filter for only 10-Q and 10-K documents
                        if '10-Q' in text or '10-K' in text:
                            match = re.search(r"(first|second|third|fourth)", quarter_name, re.IGNORECASE)
                            if match:
                                quarter_map = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
                                quarter = quarter_map[match.group(1).lower()]
                            year_links.update({
                                quarter: href,
                                # 'text': text,
                                # 'url': href
                            })
                    
                    # Close this accordion section before opening the next one
                    driver.execute_script("arguments[0].click();", button)
                    time.sleep(1)
                
                except Exception as e:
                    print(f"Error processing accordion {i}: {str(e)}")
            
            pdf_links[year] = year_links

        json_data = json.dumps(pdf_links)   
        s3_key = "metadata/metadata.json"
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=json_data,
                ContentType='application/json'
            )
            return "metadata.json file uploaded successfully "
        except:
            return "file not uploaded to s3"
        

    finally:
        # Close the browser when done
        driver.quit()


print(scrape_pdf_links())