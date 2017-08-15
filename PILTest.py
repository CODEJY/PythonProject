# -*- coding:utf-8 -*-
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import random
#随机大小写字母，数字
def randChar():
	r = random.randint(0,3);
	if r == 0:
		return chr(random.randint(48,57))
	elif r == 1:
		return chr(random.randint(65,90))
	else:
		return chr(random.randint(97,122))
#随机背景颜色
def randColor():
	return (random.randint(64,255),random.randint(64,255),random.randint(64,255))
# 随机字体颜色
def randColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

width = 240
height = 60
image = Image.new('RGB',(width,height),(255,255,255))
font = ImageFont.truetype('Arial.ttf',36)
draw = ImageDraw.Draw(image)
for x in range(width):
	for y in range(height):
		draw.point((x,y),fill=randColor())
for t in range(4):
	draw.text((60*t+10,10),randChar(),font=font,fill=randColor2())
image = image.filter(ImageFilter.BLUR)
image.save('code.jpg','jpeg')
image.show()