import RPi.GPIO as GPIO
import time

# GPIO Pin Setup for L298N Motor Driver
IN1 = 17  # Motor A
IN2 = 18
IN3 = 22  # Motor B
IN4 = 23
ENA = 24  # PWM for Motor A
ENB = 25  # PWM for Motor B

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)

# PWM setup for speed control
pwm_a = GPIO.PWM(ENA, 100)  # 100Hz frequency for Motor A
pwm_b = GPIO.PWM(ENB, 100)  # 100Hz frequency for Motor B
pwm_a.start(50)  # Start PWM with 50% duty cycle (adjust to control speed)
pwm_b.start(50)


# Movement functions
def move_forward():
    print("Moving Forward")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)


def move_left():
    print("Turning Left")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)


def move_right():
    print("Turning Right")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)


def stop():
    print("Stopping")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)


# Main test sequence
try:
    print("Testing car movement...")
    time.sleep(1)

    move_forward()
    time.sleep(2)  # Move forward for 2 seconds

    move_left()
    time.sleep(2)  # Turn left for 2 seconds

    move_right()
    time.sleep(2)  # Turn right for 2 seconds

    stop()
    time.sleep(1)  # Stop for 1 second

    print("Car movement test completed.")

except KeyboardInterrupt:
    print("Test interrupted. Stopping the car.")
    stop()

finally:
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()  # Cleanup GPIO settings
    print("GPIO cleanup done.")
