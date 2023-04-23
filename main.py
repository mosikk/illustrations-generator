import gradio as gr

from handler import handle


def main():
    with gr.Blocks() as demo:
        gr.Markdown("Generating text illustrations")
        with gr.Column(variant="panel"):
            with gr.Row(variant="compact"):
                text = gr.Textbox(
                    label="Enter your text",
                    show_label=False,
                    max_lines=10,
                    placeholder="Enter your text",
                ).style(
                    container=False,
                )
            with gr.Row(variant="compact"):
                pics_num = gr.Number(label='Number of illustrations', precision=0, value=5)
                prompt_generating_method = gr.Radio(
                    ['Summary', 'ChatGPT', 'Grammar parsing'],
                    label='Prompt generating method'
                )
                generator_type = gr.Radio(
                    ['Stable Diffusion v1.4', 'Stable Diffusion v2.1', 'Stable Diffusion with latents inheritance'],
                    label='Generator type'
                )
                guidance_scale = gr.Number(label='Guidance scale', value=7.5)
                num_inference_steps = gr.Number(label='Number of denoising iterations', precision=0, value=50)
                button = gr.Button("Generate image").style(full_width=False)

            gallery = gr.Gallery(elem_id="gallery").style(grid=[4])

        button.click(
            handle,
            inputs=[text, pics_num, prompt_generating_method, generator_type, num_inference_steps, guidance_scale],
            outputs=gallery,
        )

    demo.queue(concurrency_count=10).launch(debug=True, share=True)


if __name__ == '__main__':
    main()
