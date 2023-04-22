
import json

import requests

import gradio as gr


MODEL_PARAMS = {
    'ru-alpaca-7b-q4': {
        'temperature': 0.2,
        'max_tokens': 512,
    },
    'saiga-7b-q4': {
        'temperature': 0.2,
        'max_tokens': 2000,
    },
    'saiga-7b-v2-q4': {
        'temperature': 0.2,
        'max_tokens': 2000,
    },
}
MODELS = list(MODEL_PARAMS)

EXAMPLES = [
    'В чем основные различия между языками программирования Python и JavaScript?',
    'Что, если бы Интернет был изобретен в эпоху Возрождения?',
    'Если конечными точками отрезка прямой являются (2, -2) и (10, 4), то какова длина отрезка?',
]


def api_complete(prompt, model='saiga-7b-q4', max_tokens=128, temperature=0.2):
    response = requests.post(
        'https://api.rulm.alexkuk.ru/v1/complete',
        json={
            'prompt': prompt,
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature
        },
        stream=True
    )
    for line in response.iter_lines():
        yield json.loads(line)

        # {'n_past': 64, 'n_tokens': 78, 'text': None}
        # {'n_past': 78, 'n_tokens': 78, 'text': None}
        # {'n_past': None, 'n_tokens': None, 'text': 'В'}
        # {'n_past': None, 'n_tokens': None, 'text': 'ы'}


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(lines=20, show_label=False)
            examples = gr.Examples(
                examples=EXAMPLES,
                inputs=[text],
                label='Примеры'
            )

        with gr.Column():
            model = gr.Dropdown(MODELS, value='saiga-7b-q4', show_label=False)
            temperature = gr.Slider(
                0, 1, step=0.1, value=0.2,
                label='temperature'
            )
            max_tokens = gr.Slider(
                1, 2000, step=1, value=128,
                label='max_tokens'
            )
            submit = gr.Button('Отправить', variant='primary')
            cancel = gr.Button('Отменить', variant='secondary')

    def model_change(model):
        params = MODEL_PARAMS[model]
        return {
            temperature: gr.Slider.update(value=params['temperature']),
            max_tokens: gr.Slider.update(maximum=params['max_tokens'])
        }

    def submit_click(text, model, temperature, max_tokens, progress=gr.Progress()):
        items = api_complete(
            prompt=text, model=model,
            temperature=temperature, max_tokens=max_tokens
        )

        output = text
        for item in items:
            if item['text']:
                output += item['text']
                yield output
            else:
                progress(
                    (item['n_past'], item['n_tokens']),
                    desc='Обрабатывает промпт'
                )

    model.change(
        fn=model_change,
        inputs=[model],
        outputs=[temperature, max_tokens]
    )
    submit_click_event = submit.click(
        fn=submit_click,
        inputs=[text, model, temperature, max_tokens],
        outputs=[text]
    )
    cancel.click(fn=None, cancels=[submit_click_event])


demo.queue(
    concurrency_count=2,
    api_open=False
)
demo.launch()
