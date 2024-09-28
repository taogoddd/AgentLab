import copy
import gymnasium as gym
import logging
import numpy as np
import playwright.sync_api
import time
import re

from abc import ABC
from pathlib import Path
from typing import Optional

from .chat import Chat
from .task import AbstractBrowserTask
from .spaces import Unicode, AnyDict, AnyBox
from .constants import TEXT_MAX_LENGTH, BROWSERGYM_ID_ATTRIBUTE, EXTRACT_OBS_MAX_TRIES
from .observation import (
    _pre_extract,
    _post_extract,
    extract_screenshot,
    extract_dom_snapshot,
    extract_dom_extra_properties,
    extract_merged_axtree,
    extract_focused_element_bid,
    MarkingError,
    get_page_bboxes
)
from .action.base import execute_python_code
from .action.highlevel import HighLevelActionSet
from .action.base import execute_python_code
from . import _get_global_playwright

from browsergym.utils.obs import (
    flatten_axtree_to_str,
    search_keyword_from_tree_str,
    format_function_call_str,
)

import pandas as pd
from io import StringIO

import os

from PIL import Image, ImageDraw, ImageFont

from playwright.sync_api import CDPSession, Page, ViewportSize

import requests

import json

from typing import List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)
import torch
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

class BrowserEnv(gym.Env, ABC):
    """The main BrowserGym class, which encapsulates instruction-following Web browsing into a Gymnasium environment."""

    # gym metadata
    metadata = {"render_modes": None}

    def __init__(
        self,
        # task-related arguments
        task_entrypoint: type[AbstractBrowserTask],
        task_kwargs: dict = {},
        viewport: Optional[dict] = None,  # will override the task's viewport
        slow_mo: Optional[int] = None,  # will override the task's slow_mo
        timeout: Optional[int] = None,  # will override the task's timeout
        # interactive / debugging arguments
        headless: bool = True,
        wait_for_user_message: bool = False,
        terminate_on_infeasible: bool = True,
        resizeable_window: bool = False,
        record_video_dir: Optional[str] = None,
        pw_chromium_kwargs: dict = {},
        pw_context_kwargs: dict = {},
        # agent-related arguments
        action_mapping: Optional[callable] = HighLevelActionSet().to_python_code,
    ):
        """
        Instantiate a ready to use BrowserEnv gym environment.

        Args:
            task_entrypoint: a callable that returns a new task object from a seed. Used for creating a new task during `reset()`.
            task_kwargs: additional arguments passed to `task_entrypoint`.
            viewport: desired viewport size. This will override the value defined by the task, which might change its behaviour and difficulty. Should only be set for debugging/testing.
            slow_mo: desired slow_mo value for Playwright. This will override the value defined by the task, which might change its behaviour and difficulty. Should only be set for debugging/testing.
            timeout: desired timeout value for Playwright. This will override the value defined by the task, which might change its behaviour and difficulty. Should only be set for debugging/testing.
            headless: whether the browser should run in headless mode or not. This will affect the viewport size, which might change the behaviour and difficulty of the task. Headless mode should only be disabled for debugging/testing.
            wait_for_user_message: whether the environment should pause and wait for a user message in the chat after a new message is sent by the agent. Useful for running agents in interactive mode.
            resizeable_window: whether the browser window should be resizeable or not. This will affect the viewport size, which might change the behaviour and difficulty of the task. Should only be set for debugging/testing.
            record_video_dir: if set, indicates a directory to which viewport videos will be recorded.
            pw_chromium_kwargs: extra parameters for the playwright Browser. Should only be used for debugging/testing.
            pw_context_kwargs: extra parameters for the playwright BrowserContext. Should only be used for debugging/testing.
            action_mapping: if set, the environment will use this function to map every received action to executable Python code.

        """
        super().__init__()
        self.task_entrypoint = task_entrypoint
        self.task_kwargs = dict(**task_kwargs)
        self.viewport = viewport
        self.slow_mo = slow_mo
        self.timeout = timeout
        self.headless = headless
        self.wait_for_user_message = wait_for_user_message
        self.terminate_on_infeasible = terminate_on_infeasible
        self.resizeable_window = resizeable_window
        self.record_video_dir = record_video_dir
        self.pw_chromium_kwargs = pw_chromium_kwargs
        self.pw_context_kwargs = pw_context_kwargs
        self.action_mapping = action_mapping

        # task
        self.task = None

        # playwright
        self.browser: playwright.sync_api.Browser = None
        self.context: playwright.sync_api.BrowserContext = None
        self.page: playwright.sync_api.Page = None
        self.page_history: dict = {}

        # chat
        self.chat: Chat = None

        # observation space
        self.observation_space = gym.spaces.Dict(
            {
                "chat_messages": gym.spaces.Sequence(
                    gym.spaces.Dict(
                        {
                            "role": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                            "message": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                        }
                    )
                ),
                # TODO: this is redundant with chat messages, to be removed
                "goal": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "goal_image_urls": gym.spaces.Sequence(
                    Unicode(min_length=0, max_length=TEXT_MAX_LENGTH)
                ),
                "open_pages_urls": gym.spaces.Sequence(
                    Unicode(min_length=0, max_length=TEXT_MAX_LENGTH)
                ),
                "active_page_index": gym.spaces.Box(low=0, high=255, dtype=int),
                "url": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "screenshot": AnyBox(
                    low=0,
                    high=255,
                    shape=(-1, -1, 3),
                    dtype=np.uint8,
                ),  # swapped axes (height, width, RGB)
                "dom_object": AnyDict(),
                "axtree_object": AnyDict(),
                "extra_element_properties": AnyDict(),
                "focused_element_bid": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "last_action": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "last_action_error": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "last_action_result": Unicode(min_length=0, max_length=TEXT_MAX_LENGTH),
                "elapsed_time": gym.spaces.Box(low=0, high=np.inf, dtype=float),
            }
        )

        # action space
        self.action_space = Unicode(min_length=0, max_length=TEXT_MAX_LENGTH)

        # page recovery count
        self.page_recovery_count = 0

    def close(self):
        if self.task:
            # stop the task
            self.task.teardown()
            # close the chat
            self.chat.close()
            # close the browser context
            self.context.close()
            # close the browser
            self.browser.close()
            self.task = None

    def reset(self, seed=None, *args, **kwargs):
        new_kwargs = kwargs.copy()
        # Remove 'captioning_fn' if it exists in the kwargs
        if 'captioning_fn' in new_kwargs:
            new_kwargs.pop('captioning_fn')
        super().reset(seed=seed, *args, **new_kwargs)
        self.np_random = None  # make sure all randomness is handled by the task
        self.captioning_fn = kwargs.get("captioning_fn", None)
        if self.task:
            self.task.teardown()
            self.context.close()
            self.chat.close()
            self.browser.close()

        # create a new task
        self.task = self.task_entrypoint(seed=seed, **self.task_kwargs)

        def override_property(task, env, property):
            """Extract property value from env if not None, otherwise from task."""
            env_value = getattr(env, property)
            task_value = getattr(task, property)
            if env_value is None:
                return task_value
            else:
                logger.warning(
                    f"Overriding the task's {property} parameter ({repr(task_value)} => {repr(env_value)}). This might change the task's behaviour and difficulty."
                )
                return env_value

        # fetch task's desired parameters for browser setup
        viewport = override_property(self.task, self, "viewport")
        slow_mo = override_property(self.task, self, "slow_mo")
        timeout = override_property(self.task, self, "timeout")

        # use the global Playwright instance
        pw: playwright.sync_api.Playwright = _get_global_playwright()
        # important: change playwright's test id attribute from "data-testid" to "bid"
        pw.selectors.set_test_id_attribute(BROWSERGYM_ID_ATTRIBUTE)

        # create a new browser
        self.browser = pw.chromium.launch(
            headless=self.headless,
            slow_mo=slow_mo,
            args=(
                [f"--window-size={viewport['width']},{viewport['height']}"]
                if self.resizeable_window
                else None
            ),
            # will raise an Exception if above args are overriden
            **self.pw_chromium_kwargs,
        )

        # create a new browser context for pages
        self.context = self.browser.new_context(
            no_viewport=True if self.resizeable_window else None,
            viewport=viewport,
            record_video_dir=(
                Path(self.record_video_dir) / "task_video" if self.record_video_dir else None
            ),
            record_video_size=viewport,
            # will raise an Exception if above args are overriden
            **self.pw_context_kwargs,
        )

        # set default timeout
        self.context.set_default_timeout(timeout)

        # hack: keep track of the active page with a javascript callback
        # there is no concept of active page in playwright
        # https://github.com/microsoft/playwright/issues/2603
        self.context.expose_binding(
            "browsergym_page_activated", lambda source: self._activate_page_from_js(source["page"])
        )
        self.context.add_init_script(
            r"""
window.browsergym_page_activated();
window.addEventListener("focus", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("focusin", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("load", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("pageshow", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("mousemove", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("mouseup", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("mousedown", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("wheel", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("keyup", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("keydown", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("input", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("touchstart", () => {window.browsergym_page_activated();}, {capture: true});
window.addEventListener("touchend", () => {window.browsergym_page_activated();}, {capture: true});
document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
        window.browsergym_page_activated();
    }
}, {capture: true});
"""
        )

        # create the chat
        self.chat = Chat(
            headless=self.headless,
            chat_size=(500, max(viewport["height"], 800)),
            record_video_dir=self.record_video_dir,
        )

        # create a new page
        self.page = self.context.new_page()
        recording_start_time = time.time()

        # setup the task
        goal, task_info = self.task.setup(page=self.page, captioning_fn=self.captioning_fn)

        # initialize the chat
        self.chat.add_message(
            role="assistant",
            msg="Hi! I am your UI assistant, I can perform web tasks for you. What can I help you with?",
        )
        # if any, add the task's goal to the chat
        if goal:

            # goal is text-only
            if isinstance(goal, str):
                goal_msg = goal

            # goal is text + images
            elif isinstance(goal, dict):
                goal_msg = goal["message"]
                for image_url in goal["image_urls"]:
                    self.chat.add_message(role="user_image", msg=image_url)

            self.chat.add_message(role="user", msg=goal_msg)

        self._wait_dom_loaded()

        # after the task's setup, the active page might have changed
        # perform a safety check
        self._active_page_check()

        # init start time
        self.start_time = time.time()

        # no action yet
        self.last_action = ""
        self.last_action_error = ""
        self.last_action_result = ""
        self.infeasible_message_received = False

        # if asked, wait for user message
        self._wait_for_user_message()

        # extract obs and info from environment
        obs = self._get_obs()

        info = {}
        info["task_info"] = task_info

        # TODO this is a bit hacky, find a better solution to record videos
        if self.record_video_dir:
            info["recording_start_time"] = recording_start_time
            info["recording_file"] = str(self.page.video.path())
            info["chat"] = {
                "recording_start_time": self.chat.recording_start_time,
                "recording_file": str(self.chat.page.video.path()),
            }

        return obs, info

    def step(self, action: str) -> tuple:

        self.last_action = action

        info = {}
        info["action_exec_start"] = time.time()
        info["action_exec_timeout"] = 0

        # for webarena only
        def stop_and_output(answer: str):
            self.chat.add_message(role="assistant", msg=answer)

        def send_message_to_user(text: str):
            self.chat.add_message(role="assistant", msg=text)

        def report_infeasible_instructions(reason: str):
            self.chat.add_message(role="infeasible", msg=reason)
            self.infeasible_message_received = True

        # hack: use url to directly finish this action; should be replaced by a seq of actions
        def filter_reviews_by_keyword(keyword: str):
            page = self.page
            if keyword == "disappointed":
                page.goto("http://localhost:7780/admin/review/product/index/filter/Y3JlYXRlZF9hdCU1QmxvY2FsZSU1RD1lbl9VUyZkZXRhaWw9ZGlzYXBwb2ludGVk/internal_reviews//form_key/mSfXgxHdha42VOd2/")
            elif keyword == "satisfied":
                page.goto("http://localhost:7780/admin/review/product/index/filter/Y3JlYXRlZF9hdCU1QmxvY2FsZSU1RD1lbl9VUyZkZXRhaWw9c2F0aXNmaWVk/internal_reviews//form_key/AS9yw7kELOYn30q8/")
            elif keyword == "decent":
                page.goto("http://localhost:7780/admin/review/product/index/filter/Y3JlYXRlZF9hdCU1QmxvY2FsZSU1RD1lbl9VUyZkZXRhaWw9ZGVjZW50/internal_reviews//form_key/AS9yw7kELOYn30q8/")
            elif keyword == "not useful":
                page.goto("http://localhost:7780/admin/review/product/index/filter/Y3JlYXRlZF9hdCU1QmxvY2FsZSU1RD1lbl9VUyZkZXRhaWw9bm90K3VzZWZ1bA==/internal_reviews//form_key/AS9yw7kELOYn30q8/")
            elif keyword == "best":
                page.goto("http://localhost:7780/admin/review/product/index/filter/Y3JlYXRlZF9hdCU1QmxvY2FsZSU1RD1lbl9VUyZkZXRhaWw9YmVzdA==/internal_reviews//form_key/AS9yw7kELOYn30q8/")

        # hack: context length is not passed by agent rather it is hardcoded here
        def search_keyword(keyword: str):
            """
            Search the keyword in the ax tree of the page and return the left and right {context_len} characters of the line
            """
            page = self.page
            # get the obs of the page
            obs = self._get_obs()
            # get the ax tree of the page
            axtree = obs["axtree_object"]
            # flatten the ax tree to string
            # hack: flags are fixed. Need to be passed from outside to make it consistent with the args
            tree_str = flatten_axtree_to_str(
                AX_tree=axtree,
                extra_properties=obs["extra_element_properties"],
                with_visible=True,
                with_center_coords=False,
                with_bounding_box_coords=False,
                filter_visible_only=False,
            )

            # stringfy the function call
            params = {
                "keyword": keyword,
            }
            fc_str = format_function_call_str("search_keyword", params)

            results = search_keyword_from_tree_str(tree_str, keyword, context_len=300)

            results_str = ""

            for i, result in enumerate(results):
                results_str += f"Searching result {i+1}:\n{result}\n\n"

            last_action_result_str = f"Last action: {fc_str}\nSearching results:\nNOTE: The keyword in the results is highlight like: [{keyword.upper()}]\n{results_str}"

            self.last_action_result = last_action_result_str

        # try to execute the action
        logger.debug(f"Executing action")
        try:
            if self.action_mapping:
                code = self.action_mapping(action)
            else:
                code = action
            execute_python_code(
                code,
                self.page,
                custom_functions={
                    "send_message_to_user": send_message_to_user,
                    "report_infeasible_instructions": report_infeasible_instructions,
                    "stop_and_output": stop_and_output,
                    "search": search_keyword,
                    "filter_reviews": filter_reviews_by_keyword,
                },
            )
            self.last_action_error = ""
        except Exception as e:
            self.last_action_error = f"{type(e).__name__}: {e}"
            match = re.match("TimeoutError: Timeout ([0-9]+)ms exceeded.", self.last_action_error)
            if match:
                info["action_exec_timeout"] = float(match.groups()[0]) / 1000  # ms to sec
        logger.debug(f"Action executed")
        info["action_exec_stop"] = time.time()

        # wait a bit (for the JavaScript callback to set the active page)
        time.sleep(0.5)  # wait for JS events to be fired (half a second)
        self.context.cookies()  # trigger all waiting Playwright callbacks on the stack (hack, see https://playwright.dev/java/docs/multithreading)

        # wait for the network to idle before extracting the observation, reward etc.
        self._wait_dom_loaded()

        # after the action is executed, the active page might have changed
        # perform a safety check
        self._active_page_check()
        logger.debug(f"Active page checked")

        # if asked, wait for user message
        self._wait_for_user_message()
        logger.debug(f"User message done")

        logger.debug(f"Initiating task validation")
        # extract reward, done, user_message, info (task-specific)
        reward, done, user_message, task_info = self._task_validate()
        info["task_info"] = task_info
        logger.debug(f"Task validation done")

        # add any user message sent by the task to the chat
        if user_message:
            self.chat.add_message(role="user", msg=user_message)

        # extract observation (generic)
        obs = self._get_obs()
        logger.debug(f"Observation extracted")

        # new step API wants a 5-tuple (gymnasium)
        terminated = done or (
            self.terminate_on_infeasible and self.infeasible_message_received
        )  # task or agent can terminate the episode
        truncated = False

        return obs, reward, terminated, truncated, info

    def _task_validate(self):
        # back-up these in case validate() navigates pages and messes the history
        prev_active_page = self.page
        prev_page_history = self.page_history.copy()

        # call validate
        reward, done, user_message, info = self.task.validate(self.page, self.chat.messages)

        # safety fix, in case validate() did mess up the active page and/or page history; check up to 3 times to avoid infinite loops
        # if self.page_recovery_count > 5:
        #     self.page_recovery_count = 0
        
        if prev_active_page != self.page or prev_page_history != self.page_history:
            logger.info(
                "The active page and / or page history has changed during task.validate(). A recovery fix will be applied."
            )
            self.page = prev_active_page
            self.page_history = prev_page_history
            # self.page_recovery_count += 1

        return reward, done, user_message, info

    def _wait_for_user_message(self):
        # if last message is from the assistant, wait for a user message to continue
        # TODO: be smarter about when to wait for a user message (different action from the assistant?)
        if self.chat.messages[-1]["role"] == "assistant" and self.wait_for_user_message:
            self.chat.wait_for_user_message()

    def _wait_dom_loaded(self):
        for page in self.context.pages:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
            except playwright.sync_api.TimeoutError:
                pass
            for frame in page.frames:
                try:
                    frame.wait_for_load_state("domcontentloaded", timeout=3000)
                except playwright.sync_api.TimeoutError:
                    pass

    def _activate_page_from_js(self, page: playwright.sync_api.Page):
        logger.debug(f"_activate_page_from_js(page) called, page={str(page)}")
        if not page.context == self.context:
            raise RuntimeError(
                f"Unexpected: activating a page that belongs to a different browser context ({page})."
            )

        # add the activated page to the page history (or move it to last which is the most recent)
        if page in self.page_history:
            self.page_history[page] = self.page_history.pop(
                page
            )  # move page to the end of dictionnary
        else:
            self.page_history[page] = None  # add page to the end of dictionnary

        self.page = page

    def _active_page_check(self):
        # make sure there is always a page open
        # if all pages have been closed, create a new page
        if len(self.context.pages) == 0:
            logger.warning(f"All pages are closed, opening a new page.")
            self.page = self.context.new_page()

        # if the active page got closed, get the last active page from the history
        while self.page_history and (self.page.is_closed() or self.page not in self.context.pages):
            self.page_history.pop(self.page)  # remove active page from history
            self.page = list(self.page_history.keys())[
                -1
            ]  # set last active page as the active page (most recent)

        # active page should share the same browser context with the environment
        if self.page not in self.context.pages:
            raise RuntimeError(
                f"Unexpected: active page is not part of the browser context's open pages ({self.page})."
            )

        # active page should not be closed
        if self.page.is_closed():
            raise RuntimeError(f"Unexpected: active page has been closed ({self.page}).")
        
    def fetch_browser_info(
        self,
        page: Page,
        client: CDPSession,
    ):
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )

        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds

        # extract browser info
        win_top_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_top_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config = {
            "win_top_bound": win_top_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info = {"DOMTree": tree, "config": config}

        return info
    
    # only run this if vwa & captioning_axtree is enabled
    def get_captioned_axtree(self, page: Page, client: CDPSession):
        # get the tab info
        open_tabs = page.context.pages
        try:
            tab_titles = [tab.title() for tab in open_tabs]
            current_tab_idx = open_tabs.index(page)
            for idx in range(len(open_tabs)):
                if idx == current_tab_idx:
                    tab_titles[
                        idx
                    ] = f"Tab {idx} (current): {open_tabs[idx].title()}"
                else:
                    tab_titles[idx] = f"Tab {idx}: {open_tabs[idx].title()}"
            tab_title_str = " | ".join(tab_titles)
        except Exception:
            tab_title_str = " | ".join(
                ["Tab {idx}" for idx in range(len(open_tabs))]
            )
        
        try:
            browser_info = self.fetch_browser_info(page, client)
        except Exception:
            page.wait_for_load_state("load", timeout=2500)
            browser_info = self.fetch_browser_info(page, client)
        
        # Check if the current page is an image url
        if page.url.endswith((".jpg", ".jpeg", ".png")):
            # Load image from current url and run captioning on it.
            if page.url not in self.url2caption and self.captioning_fn is not None:
                try:
                    image = Image.open(
                        requests.get(page.url, stream=True).raw
                    )
                    caption = self.captioning_fn([image])[0].strip()
                    self.url2caption[page.url] = remove_unicode(caption)
                except Exception as e:
                    print("WARNING: ", e)

            content = self.url2caption.get(page.url, "Image")
        else:
            if self.captioning_fn is not None:
                try:
                    image_data = page.evaluate("""
                        () => {
                            const images = document.querySelectorAll('img');
                            return Array.from(images).map(img => img.getAttribute('src') || '');
                        }
                    """)
                except Exception as e:
                    print("Failed to fetch image sources: ", e)
                    image_data = []

                image_urls = []
                for image_src in image_data:
                    try:
                        if not image_src.startswith(("http://", "https://", "www.")):
                            image_src = urljoin(page.url, image_src)
                        if image_src not in self.url2caption:
                            image_urls.append(image_src)
                    except Exception as e:
                        print("WARNING:", e)

                # Run image captioning on image_url pixels. This is for models which use captioning as a baseline.
                if len(image_urls) > 0:
                    image_pixels = []
                    valid_urls = []
                    for url in image_urls:
                        if "data:image/svg" in url:
                            continue
                        else:
                            try:
                                image = Image.open(
                                    requests.get(url, stream=True).raw
                                )
                                image_pixels.append(image)
                                valid_urls.append(url)
                            except Exception as e:
                                print("WARNING: ", e)

                    # Caption images.
                    if image_pixels:
                        bs = 4
                        captions = []
                        for i in range(0, len(image_pixels), bs):
                            try:
                                captions.extend(
                                    self.captioning_fn(
                                        image_pixels[i : i + bs]
                                    )
                                )
                            except Exception as e:
                                print("WARNING: ", e)
                                captions.extend(
                                    [""] * len(image_pixels[i : i + bs])
                                )
                        assert len(valid_urls) == len(
                            captions
                        ), f"len(images)={len(valid_urls)}, len(captions)={len(captions)}"
                        for image_url, caption in zip(valid_urls, captions):
                            self.url2caption[image_url] = remove_unicode(
                                caption.strip()
                            )

                image_updates = []
                images_data = page.evaluate("""
                    () => {
                        const images = document.querySelectorAll('img');
                        return Array.from(images).map(img => ({
                            alt: img.getAttribute('alt') || '',
                            src: img.getAttribute('src')
                        }));
                    }
                """)
                for image_data in images_data:
                    try:
                        updated_alt, image_url = image_data['alt'], image_data['src']
                        if not image_url.startswith(("http://", "https://", "www.")):
                            image_url = urljoin(page.url, image_url)
                        if image_url in self.url2caption:
                            if self.url2caption[image_url] not in updated_alt:
                                if updated_alt:
                                    updated_alt = f"{updated_alt}, description: {self.url2caption[image_url]}"
                                else:
                                    updated_alt = f"description: {self.url2caption[image_url]}"
                        elif "data:image/svg" not in image_url:
                            print(f"WARNING: {image_url} not in self.url2caption")

                        if "url:" not in updated_alt:
                            updated_alt = f"{updated_alt}, url: {image_url}"

                        safe_updated_alt = json.dumps(updated_alt)
                        image_updates.append({'image_url': image_url, 'updated_alt': safe_updated_alt})
                    except Exception as e:
                        print("WARNING:", e)

                # Execute the batch update
                js_code = """
                    (image_updates => {
                        const images = document.querySelectorAll('img');
                        const urlToImageMap = {};
                        images.forEach(img => {
                            urlToImageMap[img.src] = img;
                        });

                        image_updates.forEach(update => {
                            const img = urlToImageMap[update.image_url];
                            if (img) {
                                img.alt = update.updated_alt;
                            }
                        });
                    })(%s);
                """ % json.dumps(image_updates)
                page.evaluate(js_code)
                content = ""  # Not used for SoM
    def draw_bounding_boxes(
        self,
        data_string,
        screenshot_img,
        viewport_size=None,
        add_ids=True,
        bbox_color=None,
        min_width=8,
        min_height=8,
        bbox_padding=0,
        bbox_border=2,
        plot_ids=None,
    ):
        df = pd.read_csv(StringIO(data_string), delimiter=",", quotechar='"')

        # Provide [id] textContent inputs to the model as text.
        text_content_elements = []
        text_content_text = set()  # Store text of interactable elements
        id2semantic = {}
        index = 0
        bbox_id2visid = {}

        for _, row in df.iterrows():
            if not row["Interactable"]:
                content = ""
                # Add image alt-text to the text representation.
                if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                    content += row["Alt"]
                # Add HTML textContent (if any) to the text representation.
                if pd.notna(row["TextContent"]):
                    content += (
                        row["TextContent"]
                        .strip()
                        .replace("\n", "")
                        .replace("\t", "")
                    )[
                        :200
                    ]  # Limit to 200 characters to avoid having too much text

                # Check if the text is a CSS selector
                if content and not (
                    content.startswith(".") and "{" in content
                ):
                    # Add elements which are not interactable as StaticText
                    if content not in text_content_text:
                        text_content_elements.append(
                            f"[] [StaticText] [{content}]"
                        )
                        text_content_text.add(content)
                continue
            unique_id = str(index + 1)
            bbox_id2visid[
                row["ID"]
            ] = unique_id  # map the bounding box ID to the unique character ID


    def _get_obs(self):

        for retries_left in reversed(range(EXTRACT_OBS_MAX_TRIES)):
            try:
                # pre-extraction, mark dom elements (set bid, set dynamic attributes like value and checked)
                _pre_extract(self.page)

                dom = extract_dom_snapshot(self.page)
                axtree = extract_merged_axtree(self.page)
                # som_bboxes = get_page_bboxes(self.page)
                # screenshot = extract_screenshot(self.page)
                # def override_property(task, env, property):
                #     """Extract property value from env if not None, otherwise from task."""
                #     env_value = getattr(env, property)
                #     task_value = getattr(task, property)
                #     if env_value is None:
                #         return task_value
                #     else:
                #         logger.warning(
                #             f"Overriding the task's {property} parameter ({repr(task_value)} => {repr(env_value)}). This might change the task's behaviour and difficulty."
                #         )
                #         return env_value
                # viewport = override_property(self.task, self, "viewport")
                # content_str = self.draw_bounding_boxes(som_bboxes=som_bboxes, screenshot_img=screenshot, viewport_size=viewport)
                focused_element_bid = extract_focused_element_bid(self.page)
                # extra_properties, som_axtree_str = extract_dom_extra_properties(dom)
                extra_properties = extract_dom_extra_properties(dom)
            except (playwright.sync_api.Error, MarkingError) as e:
                err_msg = str(e)
                # try to add robustness to async events (detached / deleted frames)
                if retries_left > 0 and (
                    "Frame was detached" in err_msg
                    or "Frame with the given frameId is not found" in err_msg
                    or "Execution context was destroyed" in err_msg
                    or "Frame has been detached" in err_msg
                    or "Cannot mark a child frame without a bid" in err_msg
                ):
                    logger.warning(
                        f"An error occured while extracting the dom and axtree. Retrying ({retries_left}/{EXTRACT_OBS_MAX_TRIES} tries left).\n{repr(e)}"
                    )
                    # post-extract cleanup (aria-roledescription attribute)
                    _post_extract(self.page)
                    time.sleep(0.5)
                    continue
                else:
                    raise e
            break

        # post-extraction cleanup of temporary info in dom
        _post_extract(self.page)

        # use first user message as goal, if any
        # use all user images before first user message as goal images, if any
        goal_msg = "There is no goal."
        goal_image_urls = []
        _prev_image_urls = []
        for msg in self.chat.messages:
            if msg["role"] == "user_image":
                _prev_image_urls.append(msg["message"])
            elif msg["role"] == "user":
                goal_msg = msg["message"]
                goal_image_urls = _prev_image_urls
                break
            else:
                pass

        # obs is generic to all tasks
        obs = {
            "chat_messages": copy.deepcopy(self.chat.messages),
            "goal": goal_msg,  # TODO: redundant with chat messages, to be removed?
            "goal_image_urls": goal_image_urls,
            "open_pages_urls": [page.url for page in self.context.pages],
            "active_page_index": np.asarray([self.context.pages.index(self.page)]),
            "url": self.page.url,
            "screenshot": extract_screenshot(self.page),
            "dom_object": dom,
            "axtree_object": axtree,
            "extra_element_properties": extra_properties,
            "focused_element_bid": focused_element_bid,
            "last_action": self.last_action,
            "last_action_error": self.last_action_error,
            "last_action_result": self.last_action_result,
            "elapsed_time": np.asarray([time.time() - self.start_time]),
            # "som_axtree_str": som_axtree_str, # program drafted axtree
        }

        return obs
