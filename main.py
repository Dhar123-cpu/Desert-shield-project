import requests
from datetime import datetime, timedelta
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from transformers import pipeline
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
    
class WeatherAI:
    def __init__(self):
        self.weather_analyzer = pipeline(
            "text-generation",
            model="gpt2",
            truncation=True,  
            pad_token_id=50256 
        )
    
    def generate_weather_insights(self, temp, humidity, wind_speed):
        prompt = f"""Analyze these weather conditions for Abu Dhabi:
        - Temperature: {temp}°C
        - Humidity: {humidity}%
        - Wind Speed: {wind_speed} km/h
        
        Provide:
        1. Health impact analysis
        2. 3 personalized safety recommendations
        3. Energy saving tips for these conditions
        """
        
        analysis = self.weather_analyzer(prompt, max_length=300, do_sample=True)
        return analysis[0]['generated_text']
    
    def detect_weather_anomalies(self, historical_data):
        """Use simple ML to detect unusual patterns"""
        temps = historical_data['Temperature (°C)'].values
        mean_temp = np.mean(temps)
        std_temp = np.std(temps)
        
        current_temp = temps[-1]
        z_score = (current_temp - mean_temp) / std_temp
        
        if abs(z_score) > 2:
            return f"Temperature anomaly detected! Current temp is {z_score:.1f} standard deviations from average."
        return None

class AlertOptimizer:
    def __init__(self):
        self.alert_history = []
        
    def should_send_alert(self, alert_type, current_conditions):
        """AI-powered alert throttling to avoid over-alerting"""
        # Simple implementation - can be enhanced with ML
        recent_alerts = [a for a in self.alert_history 
                       if a['type'] == alert_type 
                       and a['time'] > datetime.now() - timedelta(hours=6)]
        
        if len(recent_alerts) > 2:
            print(f"AI Alert Optimization: Suppressing {alert_type} - too many recent alerts")
            return False
        return True

    def log_alert(self, alert_type):
        self.alert_history.append({
            'type': alert_type,
            'time': datetime.now()
        })

def send_email_alert(subject, body, avg_temperature, avg_humidity, avg_wind_speed, ai_insights):
    sender_email = "dharshithgd01@gmail.com"
    receiver_email = "girik80@gmail.com"
    app_password = os.environ.get("APP_PASSWORD")

    if not app_password:
        print("Error: App Password not found in environment variables.")
        return

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email

    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px;">{subject}</h2>

            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #004080;">Current Weather Conditions</h3>
                <p><strong>Temperature:</strong> {avg_temperature:.1f}°C</p>
                <p><strong>Humidity:</strong> {avg_humidity:.1f}%</p>
                <p><strong>Wind Speed:</strong> {avg_wind_speed:.1f} km/h</p>
            </div>

            <div style="background-color: #e6f2ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #004080;">Detailed Alert Information</h3>
                <p>{body.replace('\\n', '<br>')}</p>
            </div>

            <div style="background-color: #d9e6ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #004080;">AI-Generated Insights</h3>
                <p>{ai_insights.replace('\\n', '<br>')}</p>
            </div>

            <div style="margin-top: 20px; font-size: 0.9em; color: #666;">
                <p>Regards,<br>Desert Shield AI System</p>
                <p><em>This project was made for the ADNOC EnergyAI competition by Dharshith Girikumar Dhanya from Private International English School, Abu Dhabi.</em></p>
                <p><small>Note: This information may not be 100% accurate. Please refer to official sources for critical decisions. Please contact me pn dharshithgd01@gmail.com.</small></p>
            </div>
        </div>
    </body>
    </html>
    """

    message.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            print(f"Attempting login with: {sender_email}") 
            server.login(sender_email, app_password)
            print("Login successful!")  

            print("Message:")  
            print(message.as_string())  

            server.sendmail(sender_email, receiver_email, message.as_string())
            print("Email alert sent with AI insights!")

    except smtplib.SMTPAuthenticationError:
        print("Authentication failed. Check your email and App Password.")
    except smtplib.SMTPConnectError:
        print("Failed to connect to the SMTP server. Check your network.")
    except smtplib.SMTPException as e:
        print(f"Error sending email: {e}")
        print(e)


def analyze_historical_data():
    try:
        historical_df = pd.read_csv("weather_history.csv")
        weather_ai = WeatherAI()
        anomaly = weather_ai.detect_weather_anomalies(historical_df)
        
        trends_report = f"""
        Historical Weather Trends:
        - Average Temperature: {historical_df['Temperature (°C)'].mean():.1f}°C
        - Max Temperature: {historical_df['Temperature (°C)'].max():.1f}°C
        - Humidity Range: {historical_df['Humidity (%)'].min()}% to {historical_df['Humidity (%)'].max()}%
        """
        
        if anomaly:
            trends_report += f"\n\nAI Detection: {anomaly}"
        
        return trends_report
    except FileNotFoundError:
        print("No historical data found. Starting fresh.")
        return None

# Main execution
def main():
    weather_ai = WeatherAI()
    alert_optimizer = AlertOptimizer()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 24.4539,  # Latitude for Abu Dhabi
        "longitude": 54.3773,  # Longitude for Abu Dhabi
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m",
        "timezone": "auto",
        "forecast_days": 2
    }

    # Fetch weather data
    response = requests.get(url, params=params)
    data = response.json()

    # Process weather data
    hourly_data = data["hourly"]
    time = hourly_data["time"]
    temperatures = hourly_data["temperature_2m"]
    humidities = hourly_data["relativehumidity_2m"]
    wind_speeds = hourly_data["windspeed_10m"]

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow_data = {
        "time": [t for t in time if t.startswith(tomorrow)],
        "temperature": [temp for t, temp in zip(time, temperatures) if t.startswith(tomorrow)],
        "humidity": [hum for t, hum in zip(time, humidities) if t.startswith(tomorrow)],
        "wind_speed": [wind for t, wind in zip(time, wind_speeds) if t.startswith(tomorrow)]
    }

    # Calculate averages
    avg_temperature = sum(tomorrow_data["temperature"]) / len(tomorrow_data["temperature"])
    avg_humidity = sum(tomorrow_data["humidity"]) / len(tomorrow_data["humidity"])
    avg_wind_speed = sum(tomorrow_data["wind_speed"]) / len(tomorrow_data["wind_speed"])

    print(f"\nTomorrow's Forecast for Abu Dhabi:")
    print(f"Average Temperature: {avg_temperature:.1f}°C")
    print(f"Average Humidity: {avg_humidity:.1f}%")
    print(f"Average Wind Speed: {avg_wind_speed:.1f} km/h")

    # Historical analysis
    historical_insights = analyze_historical_data()
    if historical_insights:
        print("\nHistorical Analysis:")
        print(historical_insights)

    EXTREME_HEAT_THRESHOLD = 48  
    SANDSTORM_WIND_THRESHOLD = 45
    HIGH_HUMIDITY_THRESHOLD = 60

    warnings = []

    # Generate AI insights for weather conditions
    ai_insights = weather_ai.generate_weather_insights(avg_temperature, avg_humidity, avg_wind_speed)

    if avg_temperature > EXTREME_HEAT_THRESHOLD:
        if alert_optimizer.should_send_alert("heat", avg_temperature):
            warnings.append("Extreme Heat Warning: Temperature is expected to exceed 40°C.")
            alert_optimizer.log_alert("heat")
            send_email_alert(
                "Extreme Heat Warning",
                "Temperature is expected to exceed 40°C tomorrow.\n\n"
                "Safety Recommendations:\n"
                "1. Stay indoors during peak heat hours\n"
                "2. Drink plenty of water\n"
                "3. Wear light clothing\n"
                "4. Check on vulnerable individuals",
                avg_temperature,
                avg_humidity,
                avg_wind_speed,
                ai_insights  
            )

    if avg_wind_speed > SANDSTORM_WIND_THRESHOLD:
        if alert_optimizer.should_send_alert("sandstorm", avg_wind_speed):
            warnings.append("Sandstorm Warning: High wind speeds expected.")
            alert_optimizer.log_alert("sandstorm")
            send_email_alert(
                "Sandstorm Warning",
                "High wind speeds expected tomorrow. Sandstorm likely!\n\n"
                "Safety Recommendations:\n"
                "1. Close windows and doors\n"
                "2. Wear protective mask if going outside\n"
                "3. Reduce driving speed\n"
                "4. Protect sensitive equipment",
                avg_temperature,
                avg_humidity,
                avg_wind_speed,
                ai_insights  
            )

    if avg_humidity > HIGH_HUMIDITY_THRESHOLD:
        if alert_optimizer.should_send_alert("humidity", avg_humidity):
            warnings.append("High Humidity Warning: Humidity is expected to exceed 80%.")
            alert_optimizer.log_alert("humidity")
            send_email_alert(
                "High Humidity Warning",
                "Humidity is expected to exceed 80% tomorrow.\n\n"
                "Safety Recommendations:\n"
                "1. Use dehumidifiers if available\n"
                "2. Stay hydrated\n"
                "3. Be aware of mold growth\n"
                "4. Take cool showers",
                avg_temperature,
                avg_humidity,
                avg_wind_speed,
                ai_insights  
            )

    # Display warnings
    if warnings:
        print("\nAI-Enhanced Warnings for Tomorrow:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\nNo warnings for tomorrow.")

    # Save data for future analysis
    tomorrow_df = pd.DataFrame({
        "Time": tomorrow_data["time"],
        "Temperature (°C)": tomorrow_data["temperature"],
        "Humidity (%)": tomorrow_data["humidity"],
        "Wind Speed (km/h)": tomorrow_data["wind_speed"],
        "Analysis_Date": datetime.now().strftime("%Y-%m-%d")
    })

    # Append to historical data
    try:
        historical_df = pd.read_csv("weather_history.csv")
        updated_df = pd.concat([historical_df, tomorrow_df], ignore_index=True)
    except FileNotFoundError:
        updated_df = tomorrow_df

    updated_df.to_csv("weather_history.csv", index=False)
    tomorrow_df.to_csv("tomorrow_forecast.csv", index=False)

    print("\nData saved for future AI analysis.")

if __name__ == "__main__":
    main()
    

from flask import Flask, render_template, jsonify
import requests
from datetime import datetime, timedelta
import pandas as pd
import os
import time

app = Flask(__name__)

def get_weather_prediction():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 24.4539,
        "longitude": 54.3773,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m",
        "timezone": "auto",
        "forecast_days": 2
    }

    response = requests.get(url, params=params)
    data = response.json()

    hourly_data = data["hourly"]
    time = hourly_data["time"]
    temperatures = hourly_data["temperature_2m"]
    humidities = hourly_data["relativehumidity_2m"]
    wind_speeds = hourly_data["windspeed_10m"]

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow_data = {
        "time": [t for t in time if t.startswith(tomorrow)],
        "temperature": [temp for t, temp in zip(time, temperatures) if t.startswith(tomorrow)],
        "humidity": [hum for t, hum in zip(time, humidities) if t.startswith(tomorrow)],
        "wind_speed": [wind for t, wind in zip(time, wind_speeds) if t.startswith(tomorrow)]
    }

    avg_temperature = sum(tomorrow_data["temperature"]) / len(tomorrow_data["temperature"])
    avg_humidity = sum(tomorrow_data["humidity"]) / len(tomorrow_data["humidity"])
    avg_wind_speed = sum(tomorrow_data["wind_speed"]) / len(tomorrow_data["wind_speed"])

    # Determine warnings
    warnings = []
    if avg_temperature > 40:
        warnings.append("Extreme Heat Warning")
    if avg_wind_speed > 20:
        warnings.append("Sandstorm Warning")
    if avg_humidity > 80:
        warnings.append("High Humidity Warning")

    return {
        "date": tomorrow,
        "temperature": round(avg_temperature, 1),
        "humidity": round(avg_humidity, 1),
        "wind_speed": round(avg_wind_speed, 1),
        "warnings": warnings
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/weather')
def weather_api():
    # Simulate loading for demo purposes
    time.sleep(1.5)
    return jsonify(get_weather_prediction())

if __name__ == '__main__':
    app.run(debug=True)