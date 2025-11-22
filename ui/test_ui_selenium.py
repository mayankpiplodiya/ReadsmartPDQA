# tests/ui/test_ui_selenium.py
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

BASE_URL = "http://127.0.0.1:5000"

def test_login_dashboard_workflow():
    opts = Options()
    opts.add_argument("--headless=new")  # run headless
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(BASE_URL + "/login")
        time.sleep(0.5)
        user = driver.find_element(By.NAME, "username")
        pwd = driver.find_element(By.NAME, "password")
        user.send_keys("admin")
        pwd.send_keys("admin")
        # submit (assuming a form with a submit button)
        driver.find_element(By.XPATH, "//button[@type='submit' or @type='button']").click()
        time.sleep(1)
        # after login we should be on dashboard
        assert BASE_URL + "/dashboard" in driver.current_url or "dashboard" in driver.page_source.lower()
        # Find textarea or file input and submit text
        txtarea = driver.find_element(By.NAME, "text")
        txtarea.clear()
        txtarea.send_keys("Testing from Selenium. " * 20)
        # Click submit
        driver.find_element(By.XPATH, "//button[@type='submit' or @type='button']").click()
        time.sleep(1)
        # look for some result text - summary present in page
        assert "Testing from Selenium" in driver.page_source or "summary" in driver.page_source.lower()
    finally:
        driver.quit()

