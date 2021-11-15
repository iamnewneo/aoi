import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np

### Actual Game Starts Here
import pygame
import torch
from PIL import Image
from pygame import mixer

from train_cnn import CNNModel
from train_lstm import RNNModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNModel()
# cnn_model.load_state_dict(
#     torch.load("./models/model_1_vehicle_pth.pth", map_location=torch.device(DEVICE))
# )
# cnn_model.eval()
cnn_model = torch.load(
    "./models/model_1_vehicle_pth.pth", map_location=torch.device(DEVICE)
)
cnn_model.eval()


lstm_model = RNNModel()
lstm_model.load_state_dict(
    torch.load("./models/lstm_model.pth", map_location=torch.device(DEVICE))
)
lstm_model.eval()

# import sched
flag = False
flag_fired = False
# game constants
WIDTH = 800
HEIGHT = 600

# global variables
running = True
pause_state = 0
score = 0
highest_score = 0
life = 3
kills = 0
difficulty = 1
level = 1
max_kills_to_difficulty_up = 5
max_difficulty_to_level_up = 5
initial_player_velocity = 3.0
initial_enemy_velocity = 1.0
weapon_shot_velocity = 5.0
single_frame_rendering_time = 0
total_time = 0
frame_count = 0
fps = 0

# game objects
player = type("Player", (), {})()
bullet = type("Bullet", (), {})()
enemies = []
lasers = []

# initialize pygame
pygame.init()

# Input key states (keyboard)
LEFT_ARROW_KEY_PRESSED = 0
RIGHT_ARROW_KEY_PRESSED = 0
UP_ARROW_KEY_PRESSED = 0
SPACE_BAR_PRESSED = 0
ENTER_KEY_PRESSED = 0
ESC_KEY_PRESSED = 0

# create display window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Invaders")
window_icon = pygame.image.load("res/images/alien.png")
pygame.display.set_icon(window_icon)

# game sounds
pause_sound = None
level_up_sound = None
weapon_annihilation_sound = None
game_over_sound = None

# create background
background_img = pygame.image.load("res/images/background.jpg")  # 800 x 600 px image
background_music_paths = [
    "res/sounds/Space_Invaders_Music.ogg",
    "res/sounds/Space_Invaders_Music_x2.ogg",
    "res/sounds/Space_Invaders_Music_x4.ogg",
    "res/sounds/Space_Invaders_Music_x8.ogg",
    "res/sounds/Space_Invaders_Music_x16.ogg",
    "res/sounds/Space_Invaders_Music_x32.ogg",
]


def val_pos(array):
    value = np.sum(array)
    position = np.unravel_index(np.argmax(array), array.shape)
    scaled_position = (position[1] * 73, position[0] * 98)
    return value, scaled_position


def get_value_and_scaled_pos(arr):
    value = arr.sum()
    position = np.unravel_index(np.argmax(arr), arr.shape)
    scaled_pos = (
        position[0] * (HEIGHT // 18),
        position[1] * (WIDTH // 18),
    )

    return value, scaled_pos


from dataclasses import dataclass


@dataclass
class AccuracyTrack:
    correct_detect = 0
    incorrect_detect = 0


accuracy_tracker = AccuracyTrack()


def get_running_accuracy(current_decision):
    if current_decision:
        accuracy_tracker.correct_detect += 1
    else:
        accuracy_tracker.incorrect_detect += 1

    acc = accuracy_tracker.correct_detect / (
        accuracy_tracker.correct_detect + accuracy_tracker.incorrect_detect + 1
    )
    return acc * 100.0


def is_location_accurate(e_x, p_x, delta=10):
    if p_x >= e_x - delta and p_x <= e_x + delta:
        return True
    return False


def get_spaceship_position(frame):

    enemy_reloading_panels = {
        0: 2818.206787109375,
        3: 2874.66845703125,
        7: 2827.524169921875,
        8: 3937.509033203125,
        9: 2128.130859375,
        11: 3019.009765625,
        12: 2277.173828125,
        13: 4285.21533203125,
        15: 1540.177978515625,
    }
    h = 0
    buffer = 0
    keys_enemy_reloading = list(enemy_reloading_panels.keys())

    resized_input_frame = cv2.resize(frame, (100, 100))
    temp_array = []
    temp_array.append(
        np.array(
            np.transpose(np.asarray(resized_input_frame), (2, 0, 1)), dtype=np.float32
        )
    )
    y = np.array(temp_array)
    x = torch.from_numpy(y)
    output = cnn_model(x[0].reshape(1, 3, 100, 100))[0].detach().numpy()

    for key in keys_enemy_reloading:
        output_enemy_reloading = output[0][key]
        value_enemy_reloading, position_enemy_reloading = get_value_and_scaled_pos(
            output_enemy_reloading
        )
        if value_enemy_reloading > enemy_reloading_panels[key] + buffer:
            return position_enemy_reloading

    return -1


def init_background_music():
    if difficulty == 1:
        mixer.quit()
        mixer.init()
    if difficulty <= 6:
        mixer.music.load(background_music_paths[difficulty - 1])
    else:
        mixer.music.load(background_music_paths[5])
    mixer.music.play(-1)


# create player class
class Player:
    def __init__(self, img_path, width, height, x, y, dx, dy, kill_sound_path):
        self.img_path = img_path
        self.img = pygame.image.load(self.img_path)
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.kill_sound_path = kill_sound_path
        self.kill_sound = mixer.Sound(self.kill_sound_path)

    def draw(self):
        window.blit(self.img, (self.x, self.y))


# create enemy class
class Enemy:
    def __init__(
        self, img_path, img_reloading_path, width, height, x, y, dx, dy, kill_sound_path
    ):
        self.img_path = img_path
        self.img = pygame.image.load(self.img_path)
        self.reloading = False
        self.img_reloading_path = img_reloading_path
        self.img_reloadig = pygame.image.load(self.img_reloading_path)
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.kill_sound_path = kill_sound_path
        self.kill_sound = mixer.Sound(self.kill_sound_path)
        self.flag_move = False

    def draw(self):
        if self.reloading == True:
            window.blit(self.img_reloadig, (self.x, self.y))

        else:
            window.blit(self.img, (self.x, self.y))


# create bullet class
class Bullet:
    def __init__(self, img_path, width, height, x, y, dx, dy, fire_sound_path):
        self.img_path = img_path
        self.img = pygame.image.load(self.img_path)
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.fired = False
        self.fire_sound_path = fire_sound_path
        self.fire_sound = mixer.Sound(self.fire_sound_path)

    def draw(self):
        if self.fired:
            window.blit(self.img, (self.x, self.y))


# create laser class
class Laser:
    def __init__(
        self,
        img_path,
        width,
        height,
        x,
        y,
        dx,
        dy,
        shoot_probability,
        relaxation_time,
        beam_sound_path,
    ):
        self.img_path = img_path
        self.img = pygame.image.load(self.img_path)
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.beamed = False
        self.shoot_probability = shoot_probability
        self.shoot_timer = 0
        self.relaxation_time = relaxation_time
        self.beam_sound_path = beam_sound_path
        self.beam_sound = mixer.Sound(self.beam_sound_path)

    def draw(self):
        if self.beamed:
            window.blit(self.img, (self.x, self.y))


def scoreboard():
    x_offset = 10
    y_offset = 10
    # set font type and size
    font = pygame.font.SysFont("calibre", 16)

    # render font and text sprites
    score_sprint = font.render("SCORE : " + str(score), True, (255, 255, 255))
    highest_score_sprint = font.render(
        "HI-SCORE : " + str(highest_score), True, (255, 255, 255)
    )
    level_sprint = font.render("LEVEL : " + str(level), True, (255, 255, 255))
    difficulty_sprint = font.render(
        "DIFFICULTY : " + str(difficulty), True, (255, 255, 255)
    )
    life_sprint = font.render(
        "LIFE LEFT : " + str(life) + " | " + ("@ " * life), True, (255, 255, 255)
    )
    # performance info
    fps_sprint = font.render("FPS : " + str(fps), True, (255, 255, 255))
    frame_time_in_ms = round(single_frame_rendering_time * 1000, 2)
    frame_time_sprint = font.render(
        "FT : " + str(frame_time_in_ms) + " ms", True, (255, 255, 255)
    )

    # place the font sprites on the screen
    window.blit(score_sprint, (x_offset, y_offset))
    window.blit(highest_score_sprint, (x_offset, y_offset + 20))
    window.blit(level_sprint, (x_offset, y_offset + 40))
    window.blit(difficulty_sprint, (x_offset, y_offset + 60))
    window.blit(life_sprint, (x_offset, y_offset + 80))

    window.blit(fps_sprint, (WIDTH - 80, y_offset))
    window.blit(frame_time_sprint, (WIDTH - 80, y_offset + 20))
    window.blit(frame_time_sprint, (WIDTH - 80, y_offset + 20))


def collision_check(object1, object2):
    x1_cm = object1.x + object1.width / 2
    y1_cm = object1.y + object1.width / 2
    x2_cm = object2.x + object2.width / 2
    y2_cm = object2.y + object2.width / 2
    distance = math.sqrt(math.pow((x2_cm - x1_cm), 2) + math.pow((y2_cm - y1_cm), 2))
    return distance < ((object1.width + object2.width) / 2)


def respawn(enemy_obj):
    enemy_obj.x = random.randint(0, (WIDTH - enemy_obj.width))
    enemy_obj.y = random.randint(
        ((HEIGHT / 10) * 1 - (enemy_obj.height / 2)),
        ((HEIGHT / 10) * 4 - (enemy_obj.height / 2)),
    )


def kill_enemy(player_obj, bullet_obj, enemy_obj):
    global score
    global kills
    global difficulty
    bullet_obj.fired = False
    enemy_obj.kill_sound.play()
    bullet_obj.x = player_obj.x + player_obj.width / 2 - bullet_obj.width / 2
    bullet_obj.y = player_obj.y + bullet_obj.height / 2
    score = score + 10 * difficulty * level
    kills += 1
    #     if kills % max_kills_to_difficulty_up == 0:
    #         difficulty += 1
    # #         if (difficulty == max_difficulty_to_level_up) and (life != 0):
    # #             level_up()
    #         init_background_music()
    print("Score:", score)
    print("level:", level)
    print("difficulty:", difficulty)
    respawn(enemy_obj)


def rebirth(player_obj):
    player_obj.x = (WIDTH / 2) - (player_obj.width / 2)
    player_obj.y = (HEIGHT / 10) * 9 - (player_obj.height / 2)


def gameover_screen():
    scoreboard()
    font = pygame.font.SysFont("freesansbold", 64)
    gameover_sprint = font.render("GAME OVER", True, (255, 255, 255))
    window.blit(gameover_sprint, (WIDTH / 2 - 140, HEIGHT / 2 - 32))
    pygame.display.update()

    mixer.music.stop()
    game_over_sound.play()
    time.sleep(13.0)
    mixer.quit()


def gameover():
    global running
    global score
    global highest_score

    if score > highest_score:
        highest_score = score

    # console display
    print("----------------")
    print("GAME OVER !!")
    print("----------------")
    print("you died at")
    print("Level:", level)
    print("difficulty:", difficulty)
    print("Your Score:", score)
    print("----------------")
    print("Try Again !!")
    print("----------------")
    running = False
    gameover_screen()


def kill_player(player_obj, enemy_obj, laser_obj):
    global life
    laser_obj.beamed = False
    player_obj.kill_sound.play()
    laser_obj.x = enemy_obj.x + enemy_obj.width / 2 - laser_obj.width / 2
    laser_obj.y = enemy_obj.y + laser_obj.height / 2
    life -= 1
    print("Life Left:", life)
    if life > 0:
        rebirth(player_obj)
    else:
        gameover()


def destroy_weapons(player_obj, bullet_obj, enemy_obj, laser_obj):
    bullet_obj.fired = False
    laser_obj.beamed = False
    weapon_annihilation_sound.play()
    bullet_obj.x = player_obj.x + player_obj.width / 2 - bullet_obj.width / 2
    bullet_obj.y = player_obj.y + bullet_obj.height / 2
    laser_obj.x = enemy_obj.x + enemy_obj.width / 2 - laser_obj.width / 2
    laser_obj.y = enemy_obj.y + laser_obj.height / 2


# timer = sched.scheduler(time.time, time.sleep)
#
#
# def calculate_fps(sc):
#     global frame_count
#     fps = frame_count
#     print("FPS =", fps)
#     frame_count = 0
#     timer.enter(60, 1, calculate_fps, (sc,))
#
#
# timer.enter(60, 1, calculate_fps, (timer, ))
# timer.run()


def pause_game():
    pause_sound.play()
    scoreboard()
    font = pygame.font.SysFont("freesansbold", 64)
    gameover_sprint = font.render("PAUSED", True, (255, 255, 255))
    window.blit(gameover_sprint, (WIDTH / 2 - 80, HEIGHT / 2 - 32))
    pygame.display.update()
    mixer.music.pause()


def init_game():
    global pause_sound
    global level_up_sound
    global game_over_sound
    global weapon_annihilation_sound

    pause_sound = mixer.Sound("res/sounds/pause.wav")
    level_up_sound = mixer.Sound("res/sounds/1up.wav")
    game_over_sound = mixer.Sound("res/sounds/gameover.wav")
    weapon_annihilation_sound = mixer.Sound("res/sounds/annihilation.wav")

    # player
    player_img_path = "res/images/spaceship.png"  # 64 x 64 px image
    player_width = 64
    player_height = 64
    player_x = (WIDTH / 2) - (player_width / 2)
    player_y = (HEIGHT / 10) * 9 - (player_height / 2)
    player_dx = initial_player_velocity
    player_dy = 0
    player_kill_sound_path = "res/sounds/explosion.wav"

    global player
    player = Player(
        player_img_path,
        player_width,
        player_height,
        player_x,
        player_y,
        player_dx,
        player_dy,
        player_kill_sound_path,
    )

    # bullet
    bullet_img_path = "res/images/bullet.png"  # 32 x 32 px image
    bullet_width = 32
    bullet_height = 32
    bullet_x = player_x + player_width / 2 - bullet_width / 2
    bullet_y = player_y + bullet_height / 2
    bullet_dx = 0
    bullet_dy = weapon_shot_velocity
    bullet_fire_sound_path = "res/sounds/gunshot.wav"

    global bullet
    bullet = Bullet(
        bullet_img_path,
        bullet_width,
        bullet_height,
        bullet_x,
        bullet_y,
        bullet_dx,
        bullet_dy,
        bullet_fire_sound_path,
    )

    # enemy (number of enemy = level number)
    enemy_img_path = "res/images/enemy1.jpg"  # 64 x 64 px image
    enemy_width = 64
    enemy_height = 64
    enemy_dx = initial_enemy_velocity
    enemy_dy = (HEIGHT / 10) / 2
    enemy_kill_sound_path = "res/sounds/enemykill.wav"

    # enemy reloading (number of enemy = level number)
    enemy_reloading_img_path = "res/images/enemy_reloading.jpg"  # 64 x 64 px image

    # laser beam (equals number of enemies and retains corresponding enemy position)
    laser_img_path = "res/images/beam.png"  # 24 x 24 px image
    laser_width = 24
    laser_height = 24
    laser_dx = 0
    laser_dy = weapon_shot_velocity
    shoot_probability = 0.3
    relaxation_time = 100
    laser_beam_sound_path = "res/sounds/laser.wav"

    global enemies
    global lasers

    enemies.clear()
    lasers.clear()

    for lev in range(level):
        enemy_x = random.randint(0, (WIDTH - enemy_width))
        enemy_y = random.randint(
            ((HEIGHT / 10) * 1 - (enemy_height / 2)),
            ((HEIGHT / 10) * 4 - (enemy_height / 2)),
        )
        laser_x = enemy_x + enemy_width / 2 - laser_width / 2
        laser_y = enemy_y + laser_height / 2

        enemy_obj = Enemy(
            enemy_img_path,
            enemy_reloading_img_path,
            enemy_width,
            enemy_height,
            enemy_x,
            enemy_y,
            enemy_dx,
            enemy_dy,
            enemy_kill_sound_path,
        )

        enemies.append(enemy_obj)

        laser_obj = Laser(
            laser_img_path,
            laser_width,
            laser_height,
            laser_x,
            laser_y,
            laser_dx,
            laser_dy,
            shoot_probability,
            relaxation_time,
            laser_beam_sound_path,
        )
        lasers.append(laser_obj)
        # lasers.append(laser_obj)


# init game
init_game()
init_background_music()
runned_once = False


# main game loop begins
while running:
    # start of frame timing
    start_time = time.time()

    # background
    window.fill((0, 0, 0))
    window.blit(background_img, (0, 0))

    # register events
    for event in pygame.event.get():
        # Quit Event
        if event.type == pygame.QUIT:
            running = False

        # Keypress Down Event
        if event.type == pygame.KEYDOWN:
            # Left Arrow Key down
            if event.key == pygame.K_LEFT:
                print("LOG: Left Arrow Key Pressed Down")
                LEFT_ARROW_KEY_PRESSED = 1
            # Right Arrow Key down
            if event.key == pygame.K_RIGHT:
                print("LOG: Right Arrow Key Pressed Down")
                RIGHT_ARROW_KEY_PRESSED = 1
            # Up Arrow Key down
            if event.key == pygame.K_UP:
                print("LOG: Up Arrow Key Pressed Down")
                UP_ARROW_KEY_PRESSED = 1
            # Space Bar down
            if event.key == pygame.K_SPACE:
                print("LOG: Space Bar Pressed Down")
                SPACE_BAR_PRESSED = 1
            # Enter Key down ("Carriage RETURN key" from old typewriter lingo)
            if event.key == pygame.K_RETURN:
                print("LOG: Enter Key Pressed Down")
                ENTER_KEY_PRESSED = 1
                pause_state += 1
            # Esc Key down
            if event.key == pygame.K_ESCAPE:
                print("LOG: Escape Key Pressed Down")
                ESC_KEY_PRESSED = 1
                pause_state += 1

        # Keypress Up Event
        if event.type == pygame.KEYUP:
            # Right Arrow Key up
            if event.key == pygame.K_RIGHT:
                print("LOG: Right Arrow Key Released")
                RIGHT_ARROW_KEY_PRESSED = 0
            # Left Arrow Key up
            if event.key == pygame.K_LEFT:
                print("LOG: Left Arrow Key Released")
                LEFT_ARROW_KEY_PRESSED = 0
            # Up Arrow Key up
            if event.key == pygame.K_UP:
                print("LOG: Up Arrow Key Released")
                UP_ARROW_KEY_PRESSED = 0
            # Space Bar up
            if event.key == pygame.K_SPACE:
                print("LOG: Space Bar Released")
                SPACE_BAR_PRESSED = 0
            # Enter Key up ("Carriage RETURN key" from old typewriter lingo)
            if event.key == pygame.K_RETURN:
                print("LOG: Enter Key Released")
                ENTER_KEY_PRESSED = 0
            # Esc Key up
            if event.key == pygame.K_ESCAPE:
                print("LOG: Escape Key Released")
                ESC_KEY_PRESSED = 0

    # check for pause game event
    if pause_state == 2:
        pause_state = 0
        runned_once = False
        mixer.music.unpause()
    if pause_state == 1:
        if not runned_once:
            runned_once = True
            pause_game()
        continue
    # manipulate game objects based on events and player actions
    # player spaceship movement

    # bullet firing
    if (SPACE_BAR_PRESSED or UP_ARROW_KEY_PRESSED or flag_fired) and not bullet.fired:
        flag_fired = False
        bullet.fired = True
        bullet.fire_sound.play()
        bullet.x = player.x + player.width / 2 - bullet.width / 2
        bullet.y = player.y + bullet.height / 2
    # bullet movement
    if bullet.fired:
        bullet.y -= bullet.dy

    # iter through every enemies and lasers
    for i in range(len(enemies)):
        # laser beaming
        if not lasers[i].beamed:
            lasers[i].shoot_timer += 1
            if lasers[i].shoot_timer == lasers[i].relaxation_time:
                lasers[i].shoot_timer = 0
                random_chance = random.randint(0, 100)
                if random_chance <= (lasers[i].shoot_probability * 100):
                    # enemies[i].reloading = False
                    lasers[i].beamed = True
                    lasers[i].beam_sound.play()
                    lasers[i].x = (
                        enemies[i].x + enemies[i].width / 2 - lasers[i].width / 2
                    )
                    lasers[i].y = enemies[i].y + lasers[i].height / 2
        # enemy movement
        # enemies[i].reloading = False

        enemies[i].x += enemies[i].dx * float(2 ** (difficulty - 1))

        # laser movement
        if lasers[i].beamed:
            enemies[i].reloading = True
            lasers[i].y += lasers[i].dy

    # collision check
    for i in range(len(enemies)):
        bullet_enemy_collision = collision_check(bullet, enemies[i])
        if bullet_enemy_collision:
            kill_enemy(player, bullet, enemies[i])

    for i in range(len(lasers)):
        laser_player_collision = collision_check(lasers[i], player)
        if laser_player_collision:
            kill_player(player, enemies[i], lasers[i])

    for i in range(len(enemies)):
        enemy_player_collision = collision_check(enemies[i], player)
        if enemy_player_collision:
            kill_enemy(player, bullet, enemies[i])
            kill_player(player, enemies[i], lasers[i])

    for i in range(len(lasers)):
        bullet_laser_collision = collision_check(bullet, lasers[i])
        if bullet_laser_collision:
            destroy_weapons(player, bullet, enemies[i], lasers[i])

    # boundary check: 0 <= x <= WIDTH, 0 <= y <= HEIGHT
    # player spaceship
    if player.x < 0:
        player.x = 0
    if player.x > WIDTH - player.width:
        player.x = WIDTH - player.width
    # enemy
    for enemy in enemies:
        if enemy.x <= 0:
            enemy.flag_move = False
            enemy.dx = abs(enemy.dx) * 1
            enemy.y += enemy.dy
        if enemy.x >= WIDTH - enemy.width:
            enemy.flag_move = True
            enemy.dx = abs(enemy.dx) * -1
            enemy.y += enemy.dy
    # bullet
    if bullet.y < 0:
        bullet.fired = False
        bullet.x = player.x + player.width / 2 - bullet.width / 2
        bullet.y = player.y + bullet.height / 2
    # laser
    for i in range(len(lasers)):
        if lasers[i].y > HEIGHT:
            lasers[i].beamed = False
            lasers[i].x = enemies[i].x + enemies[i].width / 2 - lasers[i].width / 2
            lasers[i].y = enemies[i].y + lasers[i].height / 2

    # create frame by placing objects on the surface
    scoreboard()

    for enemy in enemies:
        enemy.draw()
        if enemy.reloading == True:
            if enemy.flag_move == True:
                pygame.draw.rect(
                    window,
                    (255, 0, 0),
                    pygame.Rect(player.x - 50, enemy.y, player.x - 70, enemy.y - 10),
                    2,
                )
            else:
                pygame.draw.rect(
                    window,
                    (255, 0, 0),
                    pygame.Rect(player.x + 50, enemy.y, player.x + 70, enemy.y - 10),
                    2,
                )
    for laser in lasers:
        laser.draw()
    bullet.draw()
    # player.draw()

    # render the display
    returned_co_ordinates = get_spaceship_position(pygame.surfarray.array3d(window))
    if returned_co_ordinates != -1:
        is_correct = is_location_accurate(enemy.x, returned_co_ordinates[1], delta=20)
        runing_accuracy = get_running_accuracy(is_correct)
        print(f"Current Running Accuracy is: {runing_accuracy:.2f}")

    if returned_co_ordinates != -1 and player.x != returned_co_ordinates[1]:
        player.x = returned_co_ordinates[1]
        flag_fired = True

    pygame.display.update()

    for enemy in enemies:
        enemy.reloading = False
        enemy.draw()

    # pygame.display.update()
    # end of rendering, end on a frame
    frame_count += 1
    end_time = time.time()
    single_frame_rendering_time = end_time - start_time
    # fps = 1 / render_time

    total_time = total_time + single_frame_rendering_time
    if total_time >= 1.0:
        fps = frame_count
        frame_count = 0
        total_time = 0
