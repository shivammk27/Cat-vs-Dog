from fastai.vision.all import *
import gradio as gr


def is_cat(x): return x[0].isupper()


learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')


def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


#image = gr.Image(shape=(192, 192))
image = gr.Image()
label = gr.Label()
examples = ['Dog.jpg']

intf = gr.Interface(fn=classify_image, inputs=image,
                    outputs=label, examples=examples)
intf.launch(inline=False)
