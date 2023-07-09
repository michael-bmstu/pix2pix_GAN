from torchvision import transforms as T
import torch
import torch.nn as nn
from PIL import Image

from aiogram import Bot, Dispatcher, executor, types
from aiogram.utils import executor
import logging
from model import *

f2c_model = {
    "discriminator": Discriminator(),
    "generator": Generator()
}

load_model(f2c_model)

TOKEN = 'TELEGRAM BOT TOKEN (view in submit)'
# log level
logging.basicConfig(level=logging.INFO)

# bot int
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.answer("Привет!\n"
                         "Ты сегодня отлично выглядишь 😎")
    await message.answer("Пришли мне свою фотоку, и я оформлю ее в стиле комикса 😜\n"
                        "подробнее /help")

@dp.message_handler(commands=['help'])
async def process_help_cmd(message: types.Message):
    await message.answer("Для удобства мне придется обрезать фотографию до квадратного размера по центру 🔳, "
                        "чтобы я не удалил лишнего, можешь обрезать ее самостоятельно 👉👈\n"
                         "Я учился совсем немного, поэтому результат будет разрещением 128х128 🥴\n"
                         "Кстати, можешь попробовать отправить что-то кроме лица, но за качество не ручаюсь 😅")
    

@dp.message_handler(content_types=['photo'])
async def process_photo(message: types.Message):
    await message.photo[-1].download('img/face.jpg')
    await message.reply("Принял 😉\nПодожди немного)")
    image = Image.open('img/face.jpg')
    image.load()
    width, height = image.size
    pic_size = min(width, height)
    mean = [0.5] * 3
    std = [0.5] * 3
    transform_bot = T.Compose([
        T.ToTensor(),
        T.CenterCrop(pic_size),
        T.Resize(size=(128, 128), antialias=False),
        T.Normalize(mean, std)
    ])

    await message.answer("Загружаю изображение в модель...")
    face = transform_bot(image)
    face = face[None, :, :, :]
    print(face.shape)
    comic = f2c_model["generator"](face)
    comic = comic * mean[0] + std[0]
    image = T.ToPILImage()(torch.squeeze(comic))
    comic_path = 'img/comic.jpg'
    image.save(comic_path)
    with open(comic_path, 'rb') as photo:
        chat_id = message.from_user.id
        await dp.bot.send_photo(chat_id=chat_id, photo=photo)
    
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

