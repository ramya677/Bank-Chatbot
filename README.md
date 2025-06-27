import streamlit as st
import random
import sqlite3
import re
import os
import cv2
import numpy as np
import face_recognition
from PIL import Image
from twilio.rest import Client
from datetime import datetime
import base64
import smtplib
import ssl
from email.message import EmailMessage
import pytesseract
import requests
import json
import time
from gtts import gTTS  # Google Text-to-Speech
import io
import pygame  # For playing audio
