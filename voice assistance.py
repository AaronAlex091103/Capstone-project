import pyttsx3 as p
import speech_recognition as sr
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import wikipedia
import time
import requests

# Initialize Text-to-Speech Engine
engine = p.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', 180)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[19].id)


def speak(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()


# Function to fetch weather information
def get_weather(city_name):
    """Fetches weather information for the given city."""
    api_key = "<your_api_key>"  # Replace with your OpenWeatherMap API key
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        temperature = data["main"]["temp"]
        weather_description = data["weather"][0]["description"]
        weather_report = f"The current temperature in {city_name} is {temperature}°C with {weather_description}."
        return weather_report
    else:
        return "I couldn't fetch the weather information. Please try again."


# Initialize Recognizer
r = sr.Recognizer()

# Voice Assistant Greeting
speak("Hello sir, I am your voice assistant, Nova.")
speak("How are you today?")

# First user input (response to greeting)
with sr.Microphone() as source:
    r.energy_threshold = 10000
    r.adjust_for_ambient_noise(source, 1.2)
    print("Listening for your response...")
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        speak("I'm glad to hear that!")
        speak("What can I do for you, sir?")
    except sr.UnknownValueError:
        speak("I couldn't understand your response. Let's continue.")
        text = ""

# Second user input (main command)
with sr.Microphone() as source:
    r.energy_threshold = 10000
    r.adjust_for_ambient_noise(source, 1.2)
    print("Listening for your command...")
    audio = r.listen(source)
    try:
        command = r.recognize_google(audio).lower()
        print("Command received:", command)
    except sr.UnknownValueError:
        speak("I couldn't understand your command. Please try again.")
        command = ""

# Command Handling
if "information" in command:
    speak("You need information related to which topic?")
    with sr.Microphone() as source:
        r.energy_threshold = 10000
        r.adjust_for_ambient_noise(source, 1.2)
        print("Listening for the topic...")
        audio = r.listen(source)
        try:
            topic = r.recognize_google(audio)
            speak(f"Searching for {topic} on Wikipedia.")
            print(f"Searching for {topic} on Wikipedia...")
            # Fetch summary from Wikipedia
            summary = wikipedia.summary(topic, sentences=2)
            speak("Here is the summary:")
            speak(summary)
            print("Summary:", summary)
        except sr.UnknownValueError:
            speak("I couldn't understand the topic. Please try again.")
        except wikipedia.DisambiguationError:
            speak("The topic is ambiguous. Please specify further.")
        except wikipedia.PageError:
            speak("I couldn't find information on the topic. Please try another one.")

elif "play" in command and "video" in command:
    speak("You want me to play which video?")
    with sr.Microphone() as source:
        r.energy_threshold = 10000
        r.adjust_for_ambient_noise(source, 1.2)
        print("Listening for video name...")
        audio = r.listen(source)
        try:
            video_name = r.recognize_google(audio)
            speak(f"Playing {video_name} on YouTube.")
            print(f"Playing {video_name} on YouTube...")
            # Automate YouTube Search and Play using Selenium
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
            driver.get(f"https://www.youtube.com/results?search_query={video_name}")
            time.sleep(3)
            # Click on the first video
            video = driver.find_element(By.ID, 'video-title')
            video.click()
            time.sleep(20)  # Play video for 20 seconds
            driver.quit()
        except sr.UnknownValueError:
            speak("I couldn't understand the video name. Please try again.")
        except Exception as e:
            speak("An error occurred while trying to play the video.")
            print("Error:", e)

elif "weather" in command:
    speak("Please tell me the name of the city.")
    with sr.Microphone() as source:
        r.energy_threshold = 10000
        r.adjust_for_ambient_noise(source, 1.2)
        print("Listening for city name...")
        audio = r.listen(source)
        try:
            city_name = r.recognize_google(audio)
            weather_report = get_weather(city_name)
            speak(weather_report)
            print("Weather Report:", weather_report)
        except sr.UnknownValueError:
            speak("I couldn't understand the city name. Please try again.")

elif "news" in command:
    speak("Sure sir, fetching the latest news for you.")
    print("Fetching news...")
    # Example News Placeholder
    news_list = [
        "Headline 1: Global markets see an uptick today.",
        "Headline 2: New advancements in AI technology announced.",
        "Headline 3: A rare celestial event will occur tonight."
    ]
    for news in news_list:
        speak(news)
        print(news)

else:
    speak("I couldn't recognize your request. Please try again.")
