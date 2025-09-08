import schedule
import time
import requests
from datetime import datetime

def run_quarterly_rebalancing():
    """Run quarterly rebalancing process"""
    try:
        print(f"Running quarterly rebalancing at {datetime.now()}")
        
        # Generate recommendations
        gen_response = requests.post("http://localhost:8000/api/users/admin/generate-rebalance-recommendations")
        print(f"Generated recommendations: {gen_response.json()}")
        
        # Send emails
        email_response = requests.post("http://localhost:8000/api/users/admin/send-rebalance-emails")
        print(f"Sent emails: {email_response.json()}")
        
    except Exception as e:
        print(f"Error in quarterly rebalancing: {e}")

# Schedule for the first day of each quarter
# January 1, April 1, July 1, October 1
schedule.every().day.at("09:00").do(lambda: run_quarterly_rebalancing() if datetime.now().day == 1 and datetime.now().month in [1, 4, 7, 10] else None)

if __name__ == "__main__":
    print("Starting quarterly scheduler...")
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour