import pygame
import random
import math
from pygame import mixer
import time
import numpy as np
from dataclasses import dataclass

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"


# game constants
WIDTH = 800
HEIGHT = 600

# initialize pygame
pygame.init()


@dataclass
class Action:
    LEFT_ARROW_KEY_PRESSED: str = "LEFT_ARROW_KEY_PRESSED"
    RIGHT_ARROW_KEY_PRESSED: str = "RIGHT_ARROW_KEY_PRESSED"
    UP_ARROW_KEY_PRESSED: str = "UP_ARROW_KEY_PRESSED"
    DO_NOTHING: str = "DO_NOTHING"
    SPACE_BAR_PRESSED: str = "SPACE_BAR_PRESSED"
    LEFT_FIRE: str = "LEFT_FIRE"
    RIGHT_FIRE: str = "RIGHT_FIRE"


# create display window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Invaders")
window_icon = pygame.image.load("res/images/alien.png")
pygame.display.set_icon(window_icon)

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


class SpaceInvaderGame:
    def __init__(self) -> None:
        self.life = 3
        self.level = 1
        self.difficulty = 1
        self.score = 0
        self.highest_score = 0
        self.kills = 0
        self.init_background_music()
        self.pause_sound = mixer.Sound("res/sounds/pause.wav")
        self.level_up_sound = mixer.Sound("res/sounds/1up.wav")
        self.game_over_sound = mixer.Sound("res/sounds/gameover.wav")
        self.weapon_annihilation_sound = mixer.Sound("res/sounds/annihilation.wav")
        self.running = True
        self.pause_state = 0
        self.enemies = []
        self.lasers = []
        self.player = None
        self.bullet = None
        self.frame_count = 0
        self.fps = 0
        self.single_frame_rendering_time = 0
        self.max_kills_to_difficulty_up = 5
        self.max_difficulty_to_level_up = 5
        self.initial_player_velocity = 3.0
        self.initial_enemy_velocity = 1.0
        self.weapon_shot_velocity = 5.0
        self.total_time = 0
        self.action_list = [
            "LEFT_ARROW_KEY_PRESSED",
            "RIGHT_ARROW_KEY_PRESSED",
            "UP_ARROW_KEY_PRESSED",
            "DO_NOTHING",
            "LEFT_FIRE",
            "RIGHT_FIRE",
        ]

    def collision_check(self, object1, object2):
        x1_cm = object1.x + object1.width / 2
        y1_cm = object1.y + object1.width / 2
        x2_cm = object2.x + object2.width / 2
        y2_cm = object2.y + object2.width / 2
        distance = math.sqrt(
            math.pow((x2_cm - x1_cm), 2) + math.pow((y2_cm - y1_cm), 2)
        )
        return distance < ((object1.width + object2.width) / 2)

    def get_screen(self):
        surface_copy = pygame.display.get_surface().copy()
        screen_np = pygame.surfarray.pixels2d(surface_copy)
        screen_np = screen_np.astype(np.int16)
        return screen_np

    def init_background_music(self):
        if self.difficulty == 1:
            mixer.quit()
            mixer.init()
        if self.difficulty <= 6:
            mixer.music.load(background_music_paths[self.difficulty - 1])
        else:
            mixer.music.load(background_music_paths[5])
        mixer.music.play(-1)

    def respawn(self, enemy_obj):
        enemy_obj.x = random.randint(0, (WIDTH - enemy_obj.width))
        enemy_obj.y = random.randint(
            ((HEIGHT / 10) * 1 - (enemy_obj.height / 2)),
            ((HEIGHT / 10) * 4 - (enemy_obj.height / 2)),
        )
        return enemy_obj

    def level_up(self):
        self.level_up_sound.play()
        self.level += 1
        self.life += 1  # grant a life
        self.difficulty = 1  # reset difficulty
        if self.level % 3 == 0:
            self.player.dx += 1
            self.bullet.dy += 1
            self.max_difficulty_to_level_up += 1
            for each_laser in self.lasers:
                each_laser.shoot_probability += 0.1
                if each_laser.shoot_probability > 1.0:
                    each_laser.shoot_probability = 1.0
        if self.max_difficulty_to_level_up > 7:
            self.max_difficulty_to_level_up = 7

        font = pygame.font.SysFont("freesansbold", 64)
        gameover_sprint = font.render("LEVEL UP", True, (255, 255, 255))
        window.blit(gameover_sprint, (WIDTH / 2 - 120, HEIGHT / 2 - 32))
        pygame.display.update()
        self.init_game()
        time.sleep(1.0)

    def rebirth(self, player_obj):
        player_obj.x = (WIDTH / 2) - (player_obj.width / 2)
        player_obj.y = (HEIGHT / 10) * 9 - (player_obj.height / 2)
        return player_obj

    def gameover_screen(self):
        self.scoreboard()
        font = pygame.font.SysFont("freesansbold", 64)
        gameover_sprint = font.render("GAME OVER", True, (255, 255, 255))
        window.blit(gameover_sprint, (WIDTH / 2 - 140, HEIGHT / 2 - 32))
        pygame.display.update()

        mixer.music.stop()
        self.game_over_sound.play()
        time.sleep(5.0)
        mixer.quit()

    def gameover(self):
        if self.score > self.highest_score:
            self.highest_score = self.score

        # console display
        # print("----------------")
        # print("GAME OVER !!")
        # print("----------------")
        # print("Level:", self.level)
        # print("Difficulty:", self.difficulty)
        # print("Your Score:", self.score)
        # print("----------------")
        # print("Try Again !!")
        # print("----------------")
        self.running = False
        self.gameover_screen()

    def kill_enemy(self, player_obj, bullet_obj, enemy_obj):
        bullet_obj.fired = False
        enemy_obj.kill_sound.play()
        bullet_obj.x = player_obj.x + player_obj.width / 2 - bullet_obj.width / 2
        bullet_obj.y = player_obj.y + bullet_obj.height / 2
        self.bullet = bullet_obj
        self.score = self.score + 10 * self.difficulty * self.level
        self.kills += 1
        if self.kills % self.max_kills_to_difficulty_up == 0:
            self.difficulty += 1
            if (self.difficulty == self.max_difficulty_to_level_up) and (
                self.life != 0
            ):
                self.level_up()
            self.init_background_music()
        # print("Score:", self.score)
        # print("level:", self.level)
        # print("difficulty:", self.difficulty)
        enemy_obj = self.respawn(enemy_obj)
        return enemy_obj

    def kill_player(self, player_obj, enemy_obj, laser_obj):
        laser_obj.beamed = False
        player_obj.kill_sound.play()
        laser_obj.x = enemy_obj.x + enemy_obj.width / 2 - laser_obj.width / 2
        laser_obj.y = enemy_obj.y + laser_obj.height / 2
        self.life -= 1
        new_laser = laser_obj
        # print("Life Left:", self.life)
        if self.life > 0:
            new_player = self.rebirth(player_obj)
        else:
            self.gameover()
        return new_player, new_laser

    def scoreboard(self):
        x_offset = 10
        y_offset = 10
        # set font type and size
        font = pygame.font.SysFont("calibre", 16)

        # render font and text sprites
        score_sprint = font.render("SCORE : " + str(self.score), True, (255, 255, 255))
        highest_score_sprint = font.render(
            "HI-SCORE : " + str(self.highest_score), True, (255, 255, 255)
        )
        level_sprint = font.render("LEVEL : " + str(self.level), True, (255, 255, 255))
        difficulty_sprint = font.render(
            "DIFFICULTY : " + str(self.difficulty), True, (255, 255, 255)
        )
        life_sprint = font.render(
            "LIFE LEFT : " + str(self.life) + " | " + ("@ " * self.life),
            True,
            (255, 255, 255),
        )

        # performance info
        fps_sprint = font.render("FPS : " + str(self.fps), True, (255, 255, 255))
        frame_time_in_ms = round(self.single_frame_rendering_time * 1000, 2)
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

    def destroy_weapons(self, player_obj, bullet_obj, enemy_obj, laser_obj):
        bullet_obj.fired = False
        laser_obj.beamed = False
        self.weapon_annihilation_sound.play()
        bullet_obj.x = player_obj.x + player_obj.width / 2 - bullet_obj.width / 2
        bullet_obj.y = player_obj.y + bullet_obj.height / 2
        laser_obj.x = enemy_obj.x + enemy_obj.width / 2 - laser_obj.width / 2
        laser_obj.y = enemy_obj.y + laser_obj.height / 2
        return bullet_obj, laser_obj

    def pause_game(self):
        self.pause_sound.play()
        self.scoreboard()
        font = pygame.font.SysFont("freesansbold", 64)
        gameover_sprint = font.render("PAUSED", True, (255, 255, 255))
        window.blit(gameover_sprint, (WIDTH / 2 - 80, HEIGHT / 2 - 32))
        pygame.display.update()
        mixer.music.pause()

    def init_game(self):
        # player
        player_img_path = "res/images/spaceship.png"  # 64 x 64 px image
        player_width = 64
        player_height = 64
        player_x = (WIDTH / 2) - (player_width / 2)
        player_y = (HEIGHT / 10) * 9 - (player_height / 2)
        player_dx = self.initial_player_velocity
        player_dy = 0
        player_kill_sound_path = "res/sounds/explosion.wav"
        self.player = Player(
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
        bullet_dy = self.weapon_shot_velocity
        bullet_fire_sound_path = "res/sounds/gunshot.wav"

        self.bullet = Bullet(
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
        enemy_img_path = "res/images/enemy.png"  # 64 x 64 px image
        enemy_width = 64
        enemy_height = 64
        enemy_dx = self.initial_enemy_velocity
        enemy_dy = (HEIGHT / 10) / 2
        enemy_kill_sound_path = "res/sounds/enemykill.wav"

        # laser beam (equals number of enemies and retains corresponding enemy position)
        laser_img_path = "res/images/beam.png"  # 24 x 24 px image
        laser_width = 24
        laser_height = 24
        laser_dx = 0
        laser_dy = self.weapon_shot_velocity
        shoot_probability = 0.3
        relaxation_time = 100
        laser_beam_sound_path = "res/sounds/laser.wav"

        self.enemies.clear()
        self.lasers.clear()

        for lev in range(self.level):
            enemy_x = random.randint(0, (WIDTH - enemy_width))
            enemy_y = random.randint(
                ((HEIGHT / 10) * 1 - (enemy_height / 2)),
                ((HEIGHT / 10) * 4 - (enemy_height / 2)),
            )
            laser_x = enemy_x + enemy_width / 2 - laser_width / 2
            laser_y = enemy_y + laser_height / 2

            enemy_obj = Enemy(
                enemy_img_path,
                enemy_width,
                enemy_height,
                enemy_x,
                enemy_y,
                enemy_dx,
                enemy_dy,
                enemy_kill_sound_path,
            )
            self.enemies.append(enemy_obj)

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
            self.lasers.append(laser_obj)

    def step(self, action):
        reward = 0
        start_time = time.time()
        window.fill((0, 0, 0))
        window.blit(background_img, (0, 0))
        if action == Action.RIGHT_ARROW_KEY_PRESSED or action == Action.RIGHT_FIRE:
            self.player.x += self.player.dx
        elif action == Action.LEFT_ARROW_KEY_PRESSED or action == Action.LEFT_FIRE:
            self.player.x -= self.player.dx
        elif (
            action == Action.SPACE_BAR_PRESSED
            or action == Action.UP_ARROW_KEY_PRESSED
            or action == Action.LEFT_FIRE
            or action == Action.RIGHT_FIRE
        ) and not self.bullet.fired:
            self.bullet.fired = True
            self.bullet.fire_sound.play()
            self.bullet.x = (
                self.player.x + self.player.width / 2 - self.bullet.width / 2
            )
            self.bullet.y = self.player.y + self.bullet.height / 2

        if self.bullet.fired:
            self.bullet.y -= self.bullet.dy

        n_enemies = len(self.enemies)
        n_lasers = len(self.lasers)
        # iter through every enemies and lasers
        for i in range(n_enemies):
            # laser beaming
            if not self.lasers[i].beamed:
                self.lasers[i].shoot_timer += 1
                if self.lasers[i].shoot_timer == self.lasers[i].relaxation_time:
                    self.lasers[i].shoot_timer = 0
                    random_chance = random.randint(0, 100)
                    if random_chance <= (self.lasers[i].shoot_probability * 100):
                        self.lasers[i].beamed = True
                        self.lasers[i].beam_sound.play()
                        self.lasers[i].x = (
                            self.enemies[i].x
                            + self.enemies[i].width / 2
                            - self.lasers[i].width / 2
                        )
                        self.lasers[i].y = self.enemies[i].y + self.lasers[i].height / 2
            # enemy movement
            self.enemies[i].x += self.enemies[i].dx * float(2 ** (self.difficulty - 1))
            # laser movement
            if self.lasers[i].beamed:
                self.lasers[i].y += self.lasers[i].dy

        # collision check
        for i in range(n_enemies):
            bullet_enemy_collision = self.collision_check(self.bullet, self.enemies[i])
            if bullet_enemy_collision:
                new_enemy_obj = self.kill_enemy(
                    self.player, self.bullet, self.enemies[i]
                )
                self.enemies[i] = new_enemy_obj
                reward = 1

        for i in range(n_lasers):
            laser_player_collision = self.collision_check(self.lasers[i], self.player)
            if laser_player_collision:
                new_player, new_laser = self.kill_player(
                    self.player, self.enemies[i], self.lasers[i]
                )
                self.lasers[i] = new_laser
                self.player = new_player
                reward = -1

        for i in range(n_enemies):
            enemy_player_collision = self.collision_check(self.enemies[i], self.player)
            if enemy_player_collision:
                new_enemy_obj = self.kill_enemy(
                    self.player, self.bullet, self.enemies[i]
                )
                self.enemies[i] = new_enemy_obj
                new_player, new_laser = self.kill_player(
                    self.player, self.enemies[i], self.lasers[i]
                )
                self.lasers[i] = new_laser
                self.player = new_player
                reward = -1

        for i in range(n_lasers):
            bullet_laser_collision = self.collision_check(self.bullet, self.lasers[i])
            if bullet_laser_collision:
                new_bullet, new_laser = self.destroy_weapons(
                    self.player, self.bullet, self.enemies[i], self.lasers[i]
                )
                self.bullet = new_bullet
                self.lasers[i] = new_laser
                reward = 0

        # boundary check: 0 <= x <= WIDTH, 0 <= y <= HEIGHT
        # player spaceship
        if self.player.x < 0:
            self.player.x = 0
        if self.player.x > WIDTH - self.player.width:
            self.player.x = WIDTH - self.player.width
        # enemy
        for i in range(n_enemies):
            enemy = self.enemies[i]
            if enemy.x <= 0:
                enemy.dx = abs(enemy.dx) * 1
                enemy.y += enemy.dy
            if enemy.x >= WIDTH - enemy.width:
                enemy.dx = abs(enemy.dx) * -1
                enemy.y += enemy.dy
            self.enemies[i] = enemy

        # bullet
        if self.bullet.y < 0:
            self.bullet.fired = False
            self.bullet.x = (
                self.player.x + self.player.width / 2 - self.bullet.width / 2
            )
            self.bullet.y = self.player.y + self.bullet.height / 2

        # laser
        for i in range(n_lasers):
            if self.lasers[i].y > HEIGHT:
                self.lasers[i].beamed = False
                self.lasers[i].x = (
                    self.enemies[i].x
                    + self.enemies[i].width / 2
                    - self.lasers[i].width / 2
                )
                self.lasers[i].y = self.enemies[i].y + self.lasers[i].height / 2

        # create frame by placing objects on the surface
        self.scoreboard()
        for laser in self.lasers:
            laser.draw()
        for enemy in self.enemies:
            enemy.draw()
        self.bullet.draw()
        self.player.draw()

        # render the display
        pygame.display.update()

        # end of rendering, end on a frame
        self.frame_count += 1
        end_time = time.time()
        single_frame_rendering_time = end_time - start_time

        self.total_time = self.total_time + single_frame_rendering_time
        if self.total_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.total_time = 0
        done = False
        if reward == -1:
            done = True
        return reward, done


if __name__ == "__main__":
    n_actions = 100000
    space_game = SpaceInvaderGame()
    space_game.init_game()
    gameover = 0
    action_prob = [0.17, 0.23, 0.02, 0.58]

    for _ in range(n_actions):
        action_taken = np.random.choice(space_game.action_list, 1, action_prob)[0]
        # print(f"Action Take: {action_taken}")
        space_game.step(action_taken)
        if space_game.life == 0:
            gameover = 1
            break

    if not gameover:
        # print("Game Not Over")
        space_game.gameover()
