# buildozer.spec
# OCR Number Scanner - Android APK
# OCR Engine: Google ML Kit Text Recognition
# Target: Android, offline OCR model support

[app]

# Application details
title = OCR Number Scanner
package.name = ocrscanner
package.domain = org.sonalikha
version = 1.0

# Source configuration
source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,db
source.exclude_dirs = tests,bin,.buildozer,easyocr_models,venv,__pycache__,.git

# Python dependencies
requirements = python3,kivy,pyjnius,opencv,numpy,pillow

# App orientation
orientation = portrait
fullscreen = 0

# Android permissions
android.permissions = CAMERA,INTERNET

# Android SDK configuration
android.api = 33
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.archs = arm64-v8a,armeabi-v7a

# Google ML Kit Latin text recognition
android.gradle_dependencies = com.google.mlkit:text-recognition:16.0.0

# Accept SDK licenses in supported Buildozer versions
android.accept_sdk_license = True

# AndroidX support for ML Kit dependencies
android.enable_androidx = True

# Application backup
android.allow_backup = True

# Presplash
presplash_color = #070812

# Optional application icon
# icon.filename = %(source.dir)s/assets/icon.png


[buildozer]

# Logging
log_level = 2

# CI/root environment
warn_on_root = 0
