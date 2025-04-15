#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pygame
import pygame_gui
import sys
import time  # For measuring typing time


# In[1]:


# Initialize pygame
pygame.init()

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

# Function to log detailed metrics to a .txt file
def log_metrics(sentence, time_taken, mistakes, wpm, times_between_chars):
    with open("typing_metrics.txt", "a") as file:
        file.write(f"Sentence: {sentence}\n")
        file.write(f"Time Taken: {time_taken:.2f} seconds\n")
        file.write(f"Mistakes: {mistakes}\n")
        file.write(f"Words Per Minute: {wpm:.2f} WPM\n")
        file.write(f"Time Between Each Character: {times_between_chars}\n")
        file.write("-" * 40 + "\n")

# Function to show the sentence on screen
def show_text(sentence):
    """ Display the sentence to type in the center of the screen. """
    font = pygame.font.SysFont("Arial", 40)  # Use Arial or any other available font
    new_text = font.render(sentence, True, "black")
    new_text_rect = new_text.get_rect(center=(WIDTH / 2, HEIGHT / 4))
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
    if typed_text == sentence_to_type:
        print("Correct! Moving to the next sentence.")
        return True
    else:
        print("Incorrect! Try again.")
        return False

# Main function to handle user typing
def get_user_name():
    current_sentence_index = 0
    while current_sentence_index < len(sentences): 
        sentence_to_type = sentences[current_sentence_index]
        
        # Start timer to measure typing time
        start_time = time.time()
        char_times = []  # To store time between character presses
        mistakes = 0
        last_char_time = start_time  # Initialize the time for the first character

        while True:
            # Event handling and UI updates
            UI_REFRESH_RATE = CLOCK.tick(60) / 1000  # Frame rate
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return  # Break the loop to exit gracefully

                if event.type == pygame.KEYDOWN:
                    # Handle 'Enter' key (K_RETURN)
                    if event.key == pygame.K_RETURN:
                        typed_text = TEXT_INPUT.text.strip()  # Get the typed text, strip whitespace
                        print(f"User typed: {typed_text}")  # For debugging

                        # Calculate time taken for the sentence
                        time_taken = time.time() - start_time

                        # Calculate WPM
                        word_count = len(typed_text.split())
                        wpm = (word_count / time_taken) * 60  # Words per minute

                        # Use the check_input_match function to compare the text
                        if check_input_match(typed_text, sentence_to_type):
                            print("Correct! Moving to the next sentence.")  # For debugging
                            log_metrics(sentence_to_type, time_taken, mistakes, wpm, char_times)  # Log metrics to file
                            current_sentence_index += 1  # Move to the next sentence
                            TEXT_INPUT.clear()  # Clear the text input field
                            break  # Exit the loop to show the next sentence
                        else:
                            mistakes += 1  # Increment mistakes counter
                            print("Incorrect! Try again.")  # For debugging

                    # Track time between each character press
                    char_time = time.time() - last_char_time  # Time since last character
                    char_times.append(char_time)
                    last_char_time = time.time()  # Update last character time

                MANAGER.process_events(event)

            # Update the UI manager
            MANAGER.update(UI_REFRESH_RATE)

            # Fill the screen with white color (clear the screen)
            SCREEN.fill("white")

            # Show the current sentence to type
            show_text(sentence_to_type)

            # Draw the UI components (including the text input field)
            MANAGER.draw_ui(SCREEN)

            # Update the display
            pygame.display.update()

    # After all sentences are typed, show the completion message
    show_text("You have completed all sentences!")
    pygame.display.update()
    pygame.time.wait(2000)  # Wait for 2 seconds before closing
    pygame.quit()  # Quit Pygame properly

# Run the game get_user_name()
get_user_name()


# In[ ]:




