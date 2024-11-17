import os
import openai
import sys
import logging
import re
from tqdm import tqdm
import random
import time

log_file = open("output.log", "w")
sys.stdout = log_file

def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (openai.RateLimitError, openai.NotFoundError, openai.APIConnectionError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay} seconds.")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

# openai.api_key = os.environ["OPENAI_API_KEY"]
# client = openai.AzureOpenAI() if os.environ.get("AZURE_OPENAI_API_KEY") else openai.OpenAI()
client = openai.OpenAI()

@retry_with_exponential_backoff
def llm(prompt_messages):
    # response = client.completions.create(
    #   model="gpt-4-0613",
    #   prompt=prompt,
    #   temperature=0,
    #   max_tokens=100,
    #   top_p=1,
    #   frequency_penalty=0.0,
    #   presence_penalty=0.0,
    #   stop=stop
    # )
    response = client.chat.completions.create(
       model = "gpt-4o-2024-05-13",
       temperature=0.1,
       messages=prompt_messages,
    )
    return response.choices[0].message.content

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

WEBSHOP_URL = "http://localhost:4000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    # print(url)
    html = requests.get(url, proxies={"http": None, "https": None}).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 3:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info

class webshopEnv:
  def __init__(self):
    self.sessions = {}
  
  def step(self, session, action):
    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'(You have clicked {button}.)'
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session])
    if observation_:
      observation += observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done
env = webshopEnv()

# trivial search & item, choose option
prompt1 = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

# prompts for chat version model
sys_prompt = """You are a helpful assistant that can issue a few actions on a shopping website to help user complete his goal on the website.
All available actions are:
search[<query>]: this action can enter the search query to search for items on the website., e.g. search[t-shirt]
click[<button id>]: this action can click on a button with the corresponding id on the website., e.g. click[B12345] or click[Buy Now]

You will be provide with all the actions and observations in history, at each step. You should think step by step and output the action you need to take in the current step with the following format:
<think>
Think step by step. Firstly check whether the shortcuts and skills given can help. If so, refer to it. If you need to make calculations such as coordinates, write them here. Describe the effect
that your previous action had on the current content of the page.
</think>

<action>
One single action in the action space to be executed. You can only use one action at a time. Never issue a action outside the action space.
</action>

Here is an example:
-----------------
Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

<think>
I need to search for a 3 ounce bottle of bright citrus deodorant for sensitive skin, so I should use 3 ounce bright citrus deodorant sensitive skin as the query.
</think>

<action>
search[3 ounce bright citrus deodorant sensitive skin]
</action>

Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

<think>
B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.
</think>

<action>
click[B078GWRC1J]
</action>

Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

<think>
For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.
</think>

<action>
click[bright citrus]
</action>

Observation: You have clicked bright citrus. 

<think>
I have clicked bright citrus. Now I should click on 3 ounce (pack of 1) to select another attribute.
</think>

<action>
click[3 ounce (pack of 1)]
</action>

Observation: You have clicked 3 ounce (pack of 1). 

<think>
I have chosen the attributes. Now I should click on Buy Now to purchase.
</think>

<action>
click[Buy Now]
</action>

IMPORTANT rules you must follow:
1. You can only issue search action when [Search] is visible in the latest observation.
2. You can only click a button when it is visible in the latest observation.
"""


# trivial search & item, choose option
prompt1_actonly = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

def webshop_run(idx, prompt, to_print=True):
  action = 'reset'
  llm_res = ""
  message = ""
  init_prompt = prompt
  prompt = ''
  parsing_error_flag = False
  for i in range(15):
    message = ""
    try:
       res = env.step(idx, action)
       observation = res[0]
    except AssertionError as e:
        # print the error message
        print(e)
        if parsing_error_flag:
            message = "IMPORTANT: Your output format is incorrect, please make sure you output in correct format."
            parsing_error_flag = False
        else:
            if action.lower().startswith('click'):
                message = "IMPORTANT: You can not click this button. Please consider another action!"
            elif action.lower().startswith('search'):
                message = "IMPORTANT: You can not search at this page. Please consider another action!"
            else:
                message = "IMPORTANT: You can not perform this action. You can only click or search on this page. Please consider another action!"

    # if action.startswith('think'):
    #   observation = 'OK.'

    if parsing_error_flag:
       message = "IMPORTANT: Your output format is incorrect, please make sure you output in correct format."
       parsing_error_flag = False

    # integrate the message into the observation
    observation = res[0]+"\n"+message

    if to_print:
      print(f"Round {i+1}")
    #   print("# System Prompt")
    #   print(sys_prompt)
      print("# Observation")
      print(observation)
      
    #   print(f'Response: {llm_res}\nAction: {action}\nObservation: {observation}\n')
      sys.stdout.flush()
    if i:
      prompt += f' {llm_res}\nObservation: {observation}\n\n'
    else:
      prompt += f'{observation}\n\n'
    
    if res[2]:  
      return res[1]

    prompt_messages = construct_prompt_messages(sys_prompt, prompt)
    llm_res = llm(prompt_messages)
    print("# Response")
    print(llm_res)
    action = parse_html_tag_output(llm_res, tags=['think', 'action'])
    if len(action) == 0 or 'action' not in action[0]:
      parsing_error_flag = True
      action = ''
    else:
        action = action[0]['action']
    print("# Parsed Action")
    print(action)

  return 0

def construct_prompt_messages(system_prompt, user_prompt):
   return [
      {
            "role": "system",
            "content": system_prompt
      },
        {
                "role": "user",
                "content": user_prompt
        }
   ]

def parse_html_tag_output(input_string: str, tags = []):
    '''
    Parse contents that are wrapped in the tags
    Args:
    input_string: str - the string to parse
    tags: List[str] - the tags to parse

    Returns:
    List[dict] - a list of dictionaries with keys as the tags and values as the contents
    '''
    try:
        # Create a pattern dynamically
        pattern_str = r""
        for tag in tags:
            pattern_str += fr"<{tag}>(.*?)</{tag}>\s*"

        # Compile the pattern
        pattern = re.compile(pattern_str, re.DOTALL)

        # Find all matches
        matches = pattern.findall(input_string)

        # Create a list of dictionaries from the matches, remove all \n
        result = [
            {tag: match[i].replace("\n", "") for i, tag in enumerate(tags)}
            for match in matches
        ]
        return result
    except Exception as e:
        print(f"Error parsing the html tag output: {e}")
        return []


def run_episodes(prompt, n=50):
  rs = []
  cnt = 0
  for i in tqdm(range(n)):
    print('-----------------')
    print(i)
    try:
      r = webshop_run(f'fixed_{i}', prompt, to_print=True)
    except AssertionError:
      r = 0
      cnt += 1
    rs.append(r)
    if (i+1) % 1 == 0:
      r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / len(rs), cnt / len(rs)
      print(i+1, r, sr, fr)
      print('-------------')
  r, sr, fr = sum(rs) / len(rs), len([_ for _ in rs if _ == 1]) / n, cnt / n
  print(r, sr, fr)
  return rs

res1 = run_episodes(prompt1, 500)

sys.stdout = sys.__stdout__  # Reset to default if needed
log_file.close()