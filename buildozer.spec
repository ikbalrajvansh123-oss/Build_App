# buildozer.spec - Android APK build configuration
# OCR Engine: Google ML Kit Text Recognition (via pyjnius)
# Model bundled in APK (~15 MB) — works fully OFFLINE
#
# Instructions:
# 1. Install Linux or WSL (Windows Subsystem for Linux)
# 2. Install Buildozer: pip install buildozer
# 3. Build APK: buildozer android debug

[app]
# Application title (displayed on device)
title = OCR Number Scanner

# Package name (unique identifier)
package.name = ocrscanner

# Package domain
package.domain = org.sonalikha

# Source code folder
source.dir = .

# Source files include
source.include_exts = py,png,jpg,kv,atlas,db

# Exclude directories
source.exclude_dirs = tests, bin, .buildozer, easyocr_models, venv, __pycache__

# App version
version = 1.0

# Dependencies
# pyjnius  → lets Python call Java/Android APIs (ML Kit)
# opencv   → image preprocessing
# pillow   → image I/O for pytesseract desktop path
requirements = python3,kivy,pyjnius,opencv,numpy,pillow

# Orientation
orientation = portrait

# Android specific
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,INTERNET

# Google ML Kit Text Recognition (Latin, bundled model = offline, ~15 MB)
android.gradle_dependencies = com.google.mlkit:text-recognition:16.0.0

# Android API level
android.api = 33
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24

# Android architecture (arm64 = modern phones, armeabi = older phones)
android.archs = arm64-v8a, armeabi-v7a

# Allow backup
android.allow_backup = True

# Icon (custom app icon path)
# icon.filename = %(source.dir)s/assets/icon.png

# Presplash
presplash_color = #070812

# Fullscreen
fullscreen = 0

[buildozer]
# Build log level
log_level = 2

# Disable root warning prompt (needed for Colab/Docker/CI)
warn_on_root = 0
