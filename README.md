# drowsiness-ML
# ML and IoT Based Driver Drowsiness Detection System

## Project Description

Driver drowsiness is one of the major causes of road accidents worldwide. This project presents a real-time Driver Drowsiness Detection System that combines Machine Learning (ML) and Internet of Things (IoT) technologies to improve road safety.

The system continuously monitors the driver's face and eye movements through a webcam. Computer vision techniques are used to detect facial features and analyze eye closure patterns. If the driver's eyes remain closed for more than a predefined threshold time, the system identifies the condition as drowsiness.

Once drowsiness is detected, the Machine Learning module sends an alert signal to an ESP32 microcontroller through serial communication. The ESP32 then activates a buzzer to warn the driver and prevent potential accidents.

## Features

* Real-time face detection
* Real-time eye monitoring
* Drowsiness detection based on eye closure duration
* ESP32-based alert system
* Buzzer notification for driver safety
* Low-cost and easy-to-implement solution

## Technologies Used

### Machine Learning

* Python
* OpenCV
* Haar Cascade Classifier / MediaPipe Face Landmark Detection

### IoT

* ESP32
* Buzzer
* Serial Communication

### Development Tools

* VS Code
* Arduino IDE

## Working Principle

1. Webcam captures live video frames.
2. Face and eyes are detected using computer vision algorithms.
3. Eye closure duration is continuously monitored.
4. If eyes remain closed beyond the threshold time, drowsiness is detected.
5. Python sends an alert signal to ESP32.
6. ESP32 activates the buzzer to alert the driver.

## Applications

* Driver safety systems
* Smart transportation
* Fleet management
* Accident prevention systems

## Future Scope

* Cloud integration using ThingSpeak or IoT platforms
* GPS-based emergency notifications
* Mobile application support
* Deep Learning-based drowsiness detection
* Real-time analytics and monitoring dashboard

## Conclusion

The ML and IoT Based Driver Drowsiness Detection System provides a simple, cost-effective, and real-time solution for detecting driver fatigue. By integrating Machine Learning with IoT hardware, the system helps improve road safety and reduce fatigue-related accidents.
