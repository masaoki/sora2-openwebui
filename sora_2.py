"""
title: OpenAI Sora 2 Video Generator
description: Pipe Function to enable video generation
author: Masaoki Kobayashi
funding_url: FREE
version: 0.2.2
license: MIT
requirements: typing, pydantic, openai, Pillow, opencv-python
environment_variables:
disclaimer: This pipe is provided as is without any guarantees.
            Please ensure that it meets your requirements.
"""

import base64
import asyncio
import io
import re
import os
import sys
import time
import uuid
from typing import Any, List, Dict, Tuple, AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from PIL import Image, ImageFilter
import cv2
from open_webui.storage.provider import Storage
from open_webui.models.files import Files, FileForm
from open_webui.models.users import Users


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="", description="OpenAI API Key")
        DURATION: str = Field(
            default="12", description="Video Duration in Second. 4, 8, or 12."
        )
        BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="Set OpenAI or compatible endpoint base URL (e.g. https://api.openai.com/v1).",
        )

    def __init__(self):
        self.type = "manifold"
        self.name = "Sora: "
        self.valves = self.Valves()
        self.emitter: Callable[[Dict[str, str]], Awaitable[None]] | None = None
        self.remix = False
        self.vid = ""

    async def emit_status(self, message: str = "", done: bool = False):
        if self.emitter:
            print(f"Emitting status {done} {message}")
            await self.emitter(
                {
                    "type": "status",
                    "data": {
                        "description": message,
                        "hidden": False,
                        "status": "completion" if done else "in_progress",
                        "done": done,
                    },
                }
            )

    async def pipes(self) -> List[dict]:
        return [
            {"id": "sora-2l", "name": "Sora 2 Landscape"},
            {"id": "sora-2p", "name": "Sora 2 Portrait"},
            {"id": "sora-2l-pro", "name": "Sora 2 Pro Landscape"},
            {"id": "sora-2p-pro", "name": "Sora 2 Pro Portrait"},
        ]

    def _adjust_size(self, input: bytes, target_size: Tuple[int, int]) -> bytes | None:
        """
        Resize the input image to fit the requirement of Sora 2.

        Args:
            input (bytes): input image binary
            target_size (tuple[int, int]): target image size (width, height)

        Returns:
            bytes | None: output image binary
        """
        try:
            img = Image.open(io.BytesIO(input))
        except (UnidentifiedImageError, ValueError):
            print("Error: Invalid image data.")
            return None

        # Draw white background for RGBA image
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background

        original_width, original_height = img.size
        original_aspect = original_width / original_height

        target_width, target_height = target_size
        target_aspect = target_width / target_height

        # Crop an image to fit the aspect ratio of target
        if original_aspect > target_aspect:
            new_width = int(target_aspect * original_height)
            offset = (original_width - new_width) / 2
            crop_box = (offset, 0, original_width - offset, original_height)
        else:
            new_height = int(original_width / target_aspect)
            offset = (original_height - new_height) / 2
            crop_box = (0, offset, original_width, original_height - offset)

        cropped_img = img.crop(crop_box)

        # Resize image
        resized_img = cropped_img.resize(target_size, Image.Resampling.LANCZOS)

        output_buffer = io.BytesIO()
        resized_img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()

    def _extract_prompt(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Extract prompt message and one picture from the message history.

        Args:
            messages: chat history

        Returns:
            (str, str): prompt and `data:` url
        """
        text: str = ""
        image: bytes | None = None
        for msg in messages:
            content = msg.get("content")
            is_list = isinstance(content, list)
            if is_list:
                text_item = "\n".join(
                    p.get("text") for p in content if p["type"] == "text"
                )
            else:
                text_item = content

            if msg.get("role") != "user":
                match = re.search(r"<!-- sora2video:([a-z0-9_]+) -->", text_item)
                if match:
                    self.remix = True
                    self.vid = match[1]
                    print(f"Remix mode for video {self.vid}")
                continue

            text = text_item

            if is_list:
                image_here = next(
                    (
                        p.get("image_url").get("url", "")
                        for p in content
                        if p["type"] == "image_url"
                    ),
                    None,
                )
                if image_here:
                    image = image_here
            else:
                match = re.search(
                    r"!\[[^\]]*\]\((data:([^;]+);base64,([^)]+))\)", content
                )
                if match:
                    image = match[1]
                    sp = match.span(1)
                    text = content[: sp[0]] + content[sp[1] :]

        return text, image

    def _extract_frame(self, path: str, frame: int) -> bytes | None:
        """
        Extract a frame from video file.

        Args:
            path: video file path
            frame: frame position

        Returns:
            bytes | None: output image binary (jpg)

        """
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()

        if ret:
            success, encoded_image = cv2.imencode(".jpg", frame)
            cap.release()
            pil_image = Image.open(io.BytesIO(encoded_image.tobytes()))
            w, h = pil_image.size
            w //= 2
            h //= 2
            print(f"Resizing image {w}x{h}")
            pil_image = pil_image.resize((w, h), Image.Resampling.LANCZOS)
            output_buffer = io.BytesIO()
            pil_image.save(output_buffer, format="PNG")
            image_bin = output_buffer.getvalue()
            print(f"Resized to {len(image_bin)} bytes.")
            return image_bin

        cap.release()
        return None

    async def generate_video(
        self,
        prompt: str,
        model: str,
        size: Tuple[int, int],
        input_reference: bytes | None,
    ) -> Tuple[str | bytes, str | None]:
        """
        Generate a video through the OpenAI Sora 2 API.

        Args:
            prompt (str): prompt
            model (str): "sora-2" or "sora-2-pro"
            size: (int, int): width and height
            input_reference (bytes): Initial frame image (optional)

        Returns:
            str | bytes: output video bytes on success, but string for errors.
            str | None: video id for later remix, None for errors.
        """

        await self.emit_status("🖼️ Generating a video...")
        key = self.valves.OPENAI_API_KEY
        if not key:
            return "Error: Valve OPENAI_API_KEY is not set", None

        openai = AsyncOpenAI(
            api_key=key,
            base_url=getattr(self.valves, "BASE_URL", "https://api.openai.com/v1"),
        )

        try:
            if self.remix:
                video = await openai.videos.remix(video_id=self.vid, prompt=prompt)
                print(f"Remix prompt: {prompt}")
            else:
                params = {
                    "model": model,
                    "prompt": prompt,
                    "seconds": getattr(self.valves, "DURATION", "8"),
                    "size": "x".join(str(s) for s in size),
                }
                if input_reference:
                    params["input_reference"] = (
                        "image.png",
                        input_reference,
                        "image/png",
                    )
                video = await openai.videos.create(**params)
            await asyncio.sleep(30)
            while video.status in ("in_progress", "queued"):
                video = await openai.videos.retrieve(video.id)
                progress = getattr(video, "progress", 0)
                await self.emit_status(f"⏳ Progress {progress} %", done=False)
                print(f"Video generation progress {progress}")
                await asyncio.sleep(5)

            print(f"Video generation completed. {video.status}")
            if video.status == "failed":
                await self.emit_status("❌ Video generation failed", done=True)
                print(f"{model} failed.", video)
                err = "Reason unknown"
                try:
                    err = f"{video.error.code}\n{video.error.message}"
                except:
                    pass
                return f"Video generation failed.\n{err}", None

            video_body = await openai.videos.download_content(video.id, variant="video")
            print(f"Video download completed. {video.status}")
            video_bin = await video_body.aread()
            print(f"Video size {len(video_bin)} bytes.")
            return video_bin, video.id

        except Exception as e:
            await self.emit_status("❌ Video generation failed", done=True)
            return f"Error during video generation: {e}", None

    async def pipe(
        self,
        body: dict,
        __user__: Dict[str, str],
        __event_emitter__: Callable[[Dict[str, Any]], Awaitable[None]] = None,
    ) -> str:
        print(f"pipe:{__name__}", body)
        messages = body.get("messages", [])
        if any("<chat_history>" in m.get("content", "") for m in messages):
            print(f"Request looks like the followup question.")
            return ""
        user = Users.get_user_by_id(__user__["id"])
        self.emitter = __event_emitter__
        model_id = body.get("model", "sora-2").split(".")[-1]
        model = "sora-2"
        size = (1280, 720)

        if model_id == "sora-2p":
            size = (720, 1280)
        elif model_id == "sora-2l-pro":
            model = "sora-2-pro"
            size = (1792, 1024)
        elif model_id == "sora-2p-pro":
            model = "sora-2-pro"
            size = (1024, 1792)

        self.remix = False
        self.vid = ""

        prompt, image_data = self._extract_prompt(body.get("messages", []))
        if image_data:
            print(f"image_data = {image_data[:30]}")

        image_bin = None
        if image_data:
            match = re.search(";base64,(.*)$", image_data)
            print(f"Attached image {image_data[:30]}")
            if match:
                image_base = base64.b64decode(match[1])
                print(f"Before adjust: {len(image_base)}")
                image_bin = self._adjust_size(image_base, size)
                print(f"After adjust: {len(image_bin)}")

        out, vid = await self.generate_video(prompt, model, size, image_bin)
        if type(out) == str:
            return out
        else:
            print(f"Binary data {len(out)} bytes.")
            id = str(uuid.uuid4())
            filename = f"{id}.mp4"
            md = {
                "OpenWebUI-User-Email": user.email,
                "OpenWebUI-User-Id": user.id,
                "OpenWebUI-User-Name": user.name,
                "OpenWebUI-File-Id": id,
                "OpenWebUI-Video-Id": vid,
            }
            print(f"Uploading {filename}")
            contents, file_path = Storage.upload_file(io.BytesIO(out), filename, md)
            print(f"Uploaded {file_path}")
            file_item = Files.insert_new_file(
                user.id,
                FileForm(
                    **{
                        "id": id,
                        "filename": filename,
                        "path": file_path,
                        "data": {},
                        "meta": {
                            "name": filename,
                            "content_type": "video/mp4",
                            "size": len(out),
                            "data": md,
                        },
                    }
                ),
            )
            print(f"Registered {file_item.id} {vid}")
            title_image = self._extract_frame(file_path, 45)  # 1.5 sec from beginning
            print(f"Created title image {len(title_image)} bytes.")
            await self.emit_status("🎉 Video generation successful", done=True)
            if title_image:
                encoded_title_image = base64.b64encode(title_image).decode("ascii")
                return (
                    "Completed.\n[![Video](data:image/jpg;base64,"
                    f"{encoded_title_image})](/api/v1/files/{file_item.id}/content)"
                    f"\n<!-- sora2video:{vid} -->\n"
                )
            return f"Completed [Video](/api/v1/files/{file_item.id}/content)!\n<!--sora2video:{vid}-->"
