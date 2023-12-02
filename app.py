import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageFilter, ImageEnhance
import time
import io

def generate_image(text, style):
    try:
        torch.backends.cudnn.benchmark = True

        # Adjust this part based on how styles are selected in your 'diffusers' library
        if style.startswith('custom:'):
            # Handle custom style input
            text = text + "," + style
        else:
            text = text + "," + style

        pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1",
                                                       torch_dtype=torch.float16)
        pipe.to('cuda')

        # Simulate a delay to show the loading spinner
        time.sleep(3)

        image = pipe(text, guidance_scale=7.5, num_inference_steps=20).images[0]
        return image
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        return None

def apply_image_filters(image, filter_type, filter_strength):
    if filter_type == '模糊':
        return image.filter(ImageFilter.BLUR)
    elif filter_type == '锐化':
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(filter_strength)
    elif filter_type == '亮度':
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(filter_strength)
    elif filter_type == '对比度':
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(filter_strength)
    else:
        return image
# Dictionary of predefined ancient poems with names and full content
ancient_poems = {
    "庐山谣": "百川东到海，何时复西归？少壮不努力，老大徒伤悲。",
    "将进酒": "君不见黄河之水天上来，奔流到海不复回。\n君不见高堂明镜悲白发，朝如青丝暮成雪。",
    "静夜思": "床前明月光，疑是地上霜。\n举头望明月，低头思故乡。",
    # Add more poems as needed
}

def main():
    # Set page title and favicon
    st.set_page_config(
        page_title="古诗成图",
        page_icon=":art:",
        layout="wide"
    )

    # Header and instructions
    st.title("古诗成图")
    st.write("欢迎使用古诗成图应用！输入一句古诗，选择图片风格，然后点击“生成图片”按钮。")

    # Input for Poetry Text
    text = st.text_input("请输入一句古诗：", help="在这里输入您想要转化为图片的古诗.")

    # Input for Image Style
    style_options = ['古风', '插画', '油画', '自然', '现代', 'custom:自定义风格']
    style = st.selectbox("选择图片风格：", style_options, help="选择生成图片的风格.")

    # Additional Customization Options
    st.sidebar.header("图像定制选项")
    filter_type = st.sidebar.selectbox("应用滤镜：", ['无', '模糊', '锐化', '亮度', '对比度'])
    filter_strength = st.sidebar.slider("滤镜强度：", 0.1, 2.0, 1.0)

    # Display predefined ancient poems in the sidebar
    selected_poem_name = st.sidebar.selectbox("选择古诗：", list(ancient_poems.keys()), help="选择一个古诗，将其插入到输入框中.")

    # Button to expand and collapse the full content of the selected poem
    if st.sidebar.button("展开/收起古诗内容"):
        if selected_poem_name in ancient_poems:
            st.sidebar.write(f"**{selected_poem_name}**:\n{ancient_poems[selected_poem_name]}")

    # Button to insert the selected poem into the text input
    if st.sidebar.button("插入古诗"):
        text = ancient_poems[selected_poem_name]

    # Create an empty placeholder for the generated image
    image_placeholder = st.empty()

    # Button to Generate Image
    generate_button = st.button("生成图片", help="点击此按钮生成基于古诗的艺术图片.")

    # Use st.spinner to display a loading spinner
    with st.spinner("正在生成图片，请稍候..."):
        if generate_button:
            generated_image = generate_image(text, style.lower())

            if generated_image is not None:
                # Apply Customization Options
                customized_image = apply_image_filters(generated_image, filter_type, filter_strength)

                # # Display the generated text
                # st.subheader("生成的古诗文本：")
                # st.text(text)

                # Display the generated image
                image_placeholder.image(customized_image, caption=f"生成的图片", use_column_width=True)
                # Shareable link
                shareable_link = st.text_input("生成的图片链接：", value="复制此链接并分享", key="shareable_link")

                # Download Button for Image
                if st.button("下载生成的图片"):
                    # Convert the image to bytes
                    img_bytes = io.BytesIO()
                    customized_image.save(img_bytes, format='PNG')

                    # Additional logging for debugging
                    print(f"Attempting to download image. Bytes length: {len(img_bytes.getvalue())}")

                    st.download_button(
                        label="下载生成的图片",
                        data=img_bytes.getvalue(),
                        file_name=f"generated_image_{style.lower()}.png",
                        key=f"download_button_{style.lower()}"
                    )

if __name__ == "__main__":
    main()