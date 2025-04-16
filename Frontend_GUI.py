#!/usr/bin/env python
# coding: utf-8

import pygame
import pygame_gui
import sys
import time  # For measuring typing time

import pygame_gui.elements.ui_text_entry_box
import torch
import torch.nn as nn
import numpy as np

from judge.judge import Judge
from judge.judgeV2 import Judge2
from judge.judgeV3 import Judge3
from judge.judgeLSTM import JudgeLSTM

from judge.dataloaderV2 import tokenize_char

metrics_file = None
training_file = None

# Initialize pygame
pygame.init()

#GUI specific structures
names = np.array([])

# Screen setup
WIDTH, HEIGHT = 1600, 900
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Text Input")

# Clock and UI manager setup
CLOCK = pygame.time.Clock()
MANAGER = pygame_gui.UIManager((WIDTH, HEIGHT))

# Sentences to be typed by the user
sentences = [
    "The quick brown fox jumps over the lazy dog. Hello, world! How's it going?"
    # More sentences can be added here
]

# Create the text input field
TEXT_INPUT = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((350, 275), (900, 50)),
    manager=MANAGER,
    object_id="main_text_entry"
)

TEXT_INPUT_PRED = pygame_gui.elements.UITextEntryBox(
    relative_rect=pygame.Rect((350, 75), (900, 500)),  # Adjust size and position
    manager=MANAGER,
    object_id="multi_line_text_entry"
)

# Function to log detailed metrics to a .txt file
def log_metrics(sentence, time_taken, mistakes, wpm, user):
    with open(metrics_file, "a") as file:
        file.write(f"Sentence: {sentence}\n")
        file.write(f"User: {user}\n")
        file.write(f"Time Taken: {time_taken:.2f} seconds\n")
        file.write(f"Mistakes: {mistakes}\n")
        file.write(f"Words Per Minute: {wpm:.2f} WPM\n")
        file.write("-" * 40 + "\n")

def log_inputs(prev_char, curr_char, time, label):
    with open(training_file, "a") as file:
        file.write(f"{prev_char}~{curr_char}~{time}~{label}\n")

# Function to show the sentence on screen
def show_text(sentence, y_offset=0, font_size=40, color="black"):
    font = pygame.font.SysFont("Arial", font_size)  # Use Arial or any other available font
    new_text = font.render(sentence, True, color)  # Set the text color
    new_text_rect = new_text.get_rect(center=(WIDTH / 2, HEIGHT / 4 + y_offset))  # Adjust vertical position with y_offset
    SCREEN.blit(new_text, new_text_rect)

# Function to compare typed text with sentence, handling newlines and spaces
def check_input_match(typed_text, sentence_to_type):
    # Remove leading/trailing spaces and handle newlines by replacing them with a space
    typed_text = typed_text.strip().replace('\n', ' ').replace('\r', '')
    sentence_to_type = sentence_to_type.strip().replace('\n', ' ').replace('\r', '')
    
    # Optionally, convert everything to lowercase for case-insensitive comparison
    typed_text = typed_text.lower()
    sentence_to_type = sentence_to_type.lower()

    # Check for a match
    return typed_text == sentence_to_type


def get_user_name():
    global names
    typed_text = ""

    TEXT_INPUT_PRED.hide()
    TEXT_INPUT.show()
    TEXT_INPUT.set_text("")
    pygame.event.clear()

    while True:
            # Event handling and UI updates
            UI_REFRESH_RATE = CLOCK.tick(60) / 1000  # Frame rate
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)  # Break the loop to exit gracefully

                if event.type == pygame.KEYDOWN:
                    # Handle 'Enter' key (K_RETURN)
                    if event.key == pygame.K_RETURN:
                        typed_text = TEXT_INPUT.text.strip()  # Get the typed text, strip whitespace

                        if typed_text not in names:
                            names = np.append(names, typed_text)
                            return 

                MANAGER.process_events(event)

            # Update the UI manager
            MANAGER.update(UI_REFRESH_RATE)

            # Fill the screen with white color (clear the screen)
            SCREEN.fill("white")

            # Show the current sentence to type
            if typed_text in names: show_text("Name already exists. Please enter a different name.", y_offset=35, font_size=20)
            show_text("Please enter your name:")

            # Draw the UI components (including the text input field)
            MANAGER.draw_ui(SCREEN)

            # Update the display
            pygame.display.update()

# Main function to handle user typing
def display_sentences():
    current_sentence_index = 0
    TEXT_INPUT.set_text("")
    pygame.event.clear()

    while current_sentence_index < len(sentences): 
        sentence_to_type = sentences[current_sentence_index]
        
        # Start timer to measure typing time
        start_time = time.time()
        mistakes = 0
        last_char_time = None  # Initialize the time for the first character
        chars = []
        typed_text = ""
        incorrect = False

        while (check_input_match(typed_text, sentence_to_type) == False):
            # Event handling and UI updates
            UI_REFRESH_RATE = CLOCK.tick(60) / 1000  # Frame rate
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)  # Break the loop to exit gracefully

                if event.type == pygame.KEYDOWN:
                    #capture the timing each key stroke
                    if (len(chars) < len(sentence_to_type)) and (event.unicode == sentence_to_type[len(chars)]): 
                        chars.append(event.unicode)  # Append the typed character to the list
                        # Track time between each character press
                        if last_char_time == None:
                            last_char_time = time.time()
                            
                        else:
                            char_time = time.time() - last_char_time  # Time since last character
                            log_inputs(chars[-2], chars[-1], char_time, len(names) - 1)
                            last_char_time = time.time()  # Update last character time

                    elif event.key != pygame.K_BACKSPACE and event.key != pygame.K_RETURN and (len(event.unicode) > 0):
                        mistakes += 1  # Increment mistakes counter

                    # Handle 'Enter' key (K_RETURN)
                    if event.key == pygame.K_RETURN:
                        typed_text = TEXT_INPUT.text.strip()  # Get the typed text, strip whitespace

                        # Calculate time taken for the sentence
                        time_taken = time.time() - start_time

                        # Calculate WPM
                        word_count = len(typed_text.split())
                        wpm = (word_count / time_taken) * 60  # Words per minute

                        # Use the check_input_match function to compare the text
                        if check_input_match(typed_text, sentence_to_type):
                            log_metrics(sentence_to_type, time_taken, mistakes, wpm, names[-1])  # Log metrics to file
                            current_sentence_index += 1  # Move to the next sentence
                            TEXT_INPUT.clear()  # Clear the text input field
                        else: 
                            incorrect = True
                            mistakes += 1

                MANAGER.process_events(event)

                # Update the UI manager
                MANAGER.update(UI_REFRESH_RATE)

                # Fill the screen with white color (clear the screen)
                SCREEN.fill("white")

                # Show the current sentence to type
                show_text(f"User now typing: {names[-1]}\n\n", y_offset=220, font_size=20)
                show_text(sentence_to_type)
                if incorrect: show_text("Incorrect. Please try again.", y_offset=150, font_size=40, color="red")

                # Draw the UI components (including the text input field)
                MANAGER.draw_ui(SCREEN)

                # Update the display
                pygame.display.update()

def display_training():
    # Hide the text input box
    TEXT_INPUT.hide()

    # Update the UI manager
    UI_REFRESH_RATE = CLOCK.tick(60) / 1000  # Frame rate
    MANAGER.update(UI_REFRESH_RATE)

    # Fill the screen with white color (clear the screen)
    SCREEN.fill("white")

    # Show the current sentence to type
    show_text("Please wait. Training the model...", y_offset=100, font_size=60)

    # Draw the UI components (including the text input field)
    MANAGER.draw_ui(SCREEN)

    # Update the display
    pygame.display.update()

def predict(model):
    global names
    prediction = None
    prev_time = None

    return_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((700, 800), (200, 80)),  # Position and size of the button
        text="Add another user",
        manager=MANAGER
    )

    TEXT_INPUT.hide()
    TEXT_INPUT_PRED.show()
    TEXT_INPUT_PRED.set_text("")
    pygame.event.clear()

    window_buffer = [(0, 0, 0)] * model.window_size  # Initialize to window size for simplicity
    outputs = torch.zeros(1, len(names))

    while True:
        # Event handling and UI updates
        UI_REFRESH_RATE = CLOCK.tick(60) / 1000  # Frame rate
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)  # Break the loop to exit gracefully

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == return_button:
                        return  # Exit the predict function

            if event.type == pygame.KEYDOWN:
                if (TEXT_INPUT_PRED.get_text() == ""): # If the text input is empty, reset the outputs
                    outputs = torch.zeros(1, len(names)) # Reset the outputs if the text input is empty
                
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)  # Break the loop to exit gracefully

                if (len(event.unicode) > 0):
                    if (prev_time == None): # If this is the first key pressed
                        prev_time = time.time()
                    else: # Otherwise...
                        curr_time = time.time()
                        letter = event.unicode
                        # Tokenize the input and add it to the buffer
                        if len(letter) > 0:  # Ignore special keys like Shift
                            tokenized_input = (tokenize_char(letter), tokenize_char(letter), curr_time - prev_time)
                            window_buffer.append(tokenized_input)

                            window_buffer.pop(0) # discard the oldest in the window buffer

                            inputs = torch.tensor(window_buffer, dtype=torch.float32).unsqueeze(0) # feed into model
                            outputs += model(inputs)
                            #print(outputs)
                            _, max_index = torch.max(outputs, 1)# get the prediction
                            prediction = names[max_index.item()] # get the index of the max value

                        prev_time = curr_time

            MANAGER.process_events(event)
        # Update the UI manager
        MANAGER.update(UI_REFRESH_RATE)

        # Fill the screen with white color (clear the screen)
        SCREEN.fill("white")

        # Show the current sentence to type
        show_text("Begin typing for predictions to appear", y_offset=-175, font_size=20)
        if (prediction != None):
            show_text(f"Prediction: {prediction}", y_offset=450, font_size=40)

        # Draw the UI components (including the text input field)
        MANAGER.draw_ui(SCREEN)

        # Update the display
        pygame.display.update()




if __name__ == "__main__":
    metrics_file = "data/typing_metrics.txt"
    training_file = "data/training.txt"

    open(metrics_file, "w").close()  # Clear the metrics file
    open(training_file, "w").close()  # Clear the training data file

    #get the first user
    get_user_name()
    display_sentences()

    while True:
        #get an additional user
        get_user_name()
        display_sentences()

        #initialize the model
        display_training()
        model = JudgeLSTM(num_classes=len(names), hidden_dim=64, window_size=3, lstm_layers=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-1)

        #train the model
        print("Trining the model:")
        model.train_model(training_file, criterion, optimizer, wandb_plot=False, random=True)

        #generate predictions
        predict(model)

