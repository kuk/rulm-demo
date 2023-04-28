
import os
import json

import requests

import gradio as gr


HOST = os.getenv('HOST', 'localhost')
PORT = int(os.getenv('PORT', 8080))

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
    'Как улучшить свои навыки управления временем (time management)?',
    'Как развить навыки критического мышления? Приведи пять пунктов:',
    'Напиши функцию на Python которая находит самую длинную общую подпоследовательность',
    'Если конечными точками отрезка прямой являются (2, -2) и (10, 4), то какова длина отрезка?',
    'Напиши регулярное выражение в Python которое проверяет адрес электронной почты',
    'Напиши программу для нахождения n-го числа Фибоначчи',
    'Реализуй алгоритм двоичного поиска в отсортированном массиве на Python',
]


class ApiError(Exception):
    pass


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
    if response.status_code != 200:
        raise ApiError(response.text)

    for line in response.iter_lines():
        item = json.loads(line)
        error = item.get('error')
        if error:
            raise ApiError(error)
        yield item


with gr.Blocks(title='Демо-стенд для русских Instruct-моделей') as demo:
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(lines=30, show_label=False)
            examples = gr.Examples(
                examples=EXAMPLES,
                inputs=[text],
                label='Примеры'
            )

        with gr.Column():
            model = gr.Dropdown(MODELS, value='saiga-7b-q4', show_label=False)
            temperature = gr.Slider(
                0, 1, value=0.2,
                label='temperature'
            )
            max_tokens = gr.Slider(
                1, 2000, step=1, value=256,
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
        try:
            items = api_complete(
                prompt=text, model=model,
                temperature=temperature, max_tokens=max_tokens
            )
            output = text + '\n'
            for item in items:
                text = item.get('text')
                prompt_progress = item.get('prompt_progress')
                if text:
                    output += text
                    yield output
                else:
                    progress(
                        prompt_progress,
                        desc='Обрабатывает промпт'
                    )
        except ApiError as error:
            raise gr.Error(str(error))

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
demo.launch(
    server_name=HOST,
    server_port=PORT
)
