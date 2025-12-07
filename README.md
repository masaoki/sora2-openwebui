Sora 2 video generation pipe function for Open WebUI.

1. Set the OPENAI_API_KEY valve. Generated video duration should be one of 4, 8, and 12 seconds by specifying DURATION valve.
1. Choose your favorite model from "Sora 2 Landscape", "Sora 2 Portrait", "Sora 2 Pro Landscape", and "Sora 2 Pro Portrait".
1. Just writing a prompt is enough to generate your video. Since the guardrails are strong, you had better refer to the OpenAI Guide page.
1. Attach an image to your prompt. The image is automatically resized to fit the Sora 2 requirement. It is used as a first frame. Video generation will fail with the people face or human-like objects because of the OpenAI guardrails.
1. Continue your conversation. Your message is used to modify the video generated with the remix API.
