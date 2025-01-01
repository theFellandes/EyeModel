import torch
import torchvision.transforms as transforms
from PIL import Image
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library

# Define model class (same as trained model)
class EyePositionCNN(torch.nn.Module):
    def __init__(self):
        super(EyePositionCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),

            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),

            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.4),

            torch.nn.Conv2d(128, 256, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.4)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 2 * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 4)  # 4 classes: close, forward, left, right
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the model
model = EyePositionCNN()
model.load_state_dict(torch.load("final_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define transformations (same as during training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define classes
classes = ["close", "forward", "left", "right", "default"]

def classify_with_default(output_probs, threshold=0.5, default_label="default"):
    """Classify based on a threshold; return default_label if no class is confident."""
    max_prob, predicted_class = torch.max(output_probs, dim=0)
    if max_prob < threshold:
        return default_label
    return classes[predicted_class.item()]

# GPIO Pin Setup for L298N Motor Driver
IN1 = 17
IN2 = 18
IN3 = 22
IN4 = 23
ENA = 24
ENB = 25

GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)

pwm_a = GPIO.PWM(ENA, 100)  # PWM for motor speed control
pwm_b = GPIO.PWM(ENB, 100)
pwm_a.start(50)  # 50% duty cycle
pwm_b.start(50)

# Movement Functions
def move_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)


def move_left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)


def move_right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)


def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Setup Raspberry Pi camera
camera = Picamera2()
camera.configure(camera.create_still_configuration({"size": (320, 240)}))  # Use lower resolution
camera.start()

# Continuously capture and classify images
print("Starting real-time car control based on eye position...")
try:
    while True:
        try:
            # Capture an image
            image_path = "current_frame.jpg"
            camera.capture_file(image_path)

            # Load the image and preprocess
            image = Image.open(image_path)
            image = transform(image).unsqueeze(0)  # Add batch dimension

            # Run the model
            with torch.no_grad():
                outputs = model(image)
                probabilities = outputs.softmax(dim=1).squeeze(0)
                prediction = classify_with_default(probabilities)

            # Move car based on prediction
            if prediction == "forward":
                move_forward()
                print("Car moving forward")
            elif prediction == "left":
                move_left()
                print("Car turning left")
            elif prediction == "right":
                move_right()
                print("Car turning right")
            elif prediction == "close":
                stop()
                print("Car stopping (eyes closed)")
            else:
                stop()
                print("Car stopping (default)")

            # Wait for 100ms before capturing the next image
            time.sleep(0.1)
        except Exception as e:
            print(f"Error processing frame: {e}")
            stop()
            time.sleep(0.5)  # Pause before retrying to avoid cascading errors
except KeyboardInterrupt:
    print("Stopping car control.")
    stop()
finally:
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()
    camera.stop()
    camera.close()
