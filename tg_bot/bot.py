from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import logging
from PIL import Image
import numpy as np
from torchvision import transforms as T
from model import *

f2c_model = {
    "discriminator": Discriminator(),
    "generator": Generator()
}

load_model(f2c_model, path_to_disc_weigths, path_to_gen_weigths)

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
    await message.answer("Пришли мне свой портрет, и я оформлю его в стиле комикса 😜\n"
                        "подробнее: /help")

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
    
    left = (width - pic_size) / 2
    top = (height - pic_size) / 2
    right = (width + pic_size) / 2
    bottom = (height + pic_size) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    arr_image = np.array(image) / 255.0

    # Dataset stats
    mean_ds = np.array([0.02437675, -0.17417155, -0.26729974])
    std_ds = np.array([0.47649799, 0.39678715, 0.37886345])

    # Image stats
    mean_img = arr_image.mean(axis=(0, 1))
    std_img = arr_image.std(axis=(0, 1))

    mean = mean_img - mean_ds
    std = std_img / std_ds
    transform_bot = T.Compose([
        T.ToTensor(),
        T.CenterCrop(pic_size),
        T.Resize(size=(128, 128), antialias=False),
        T.Normalize(mean, std)
    ])

    await message.answer("Загружаю изображение в модель...")
    face = transform_bot(image)
    face = face[None, :, :, :]
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

