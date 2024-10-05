prompt = """
You will be given a goal of a task to be executed on a website and a list of skills to choose from.
You need to select the skills that can help most in achieving the goal.

You should break the task down into a few steps so that you can select the skills that can help most in each step.
IMPORTANT: You should select not more than {max_skills} skills!

Output format:
<think>
think step by step. Break the task down into a few steps and then select skills
</think>
<selected-skills>
id: {{the id number (the number at the beginning) of skill 1}}; name: skill 1 name
id: {{the id number (the number at the beginning) of skill 2}}; name: skill 2 name
...
</selected-skills>

# Examples
## Example 1

Task goal: Upvote the hottest post in r/books
Current website: {get_website_description("reddit")}
Skills to choose from:
Skill 1: Navigate to forums
1. Click on the "Forums" menu item.
```click({{forums id}})```
2. Click on the specific forum name.
```click({{forum name id}})```
Skill 2: Submit a new post
1. Type the post title in the title text box.
```type({{title text box id}}, "Post Title")```
2. Type the post content in the content text box.
```type({{content text box id}}, "Post Content")```
3. Click on the "Submit" button.`
``click({{submit button id}})```
Skill 3: Sort posts by {{sort criterion}}
1. Click on the "Sort by" dropdown menu.
```click({{sort by dropdown id}})```
2. Select the {{sort criterion}} option from the "Sort by" dropdown menu.
```click({{sort criterion id}})```

Output:
<think>
The goal is to upvote the hottest post in r/books. The user needs to navigate to the r/books page first or go to forums to find the r/books page. Then the user needs to find the hottest post in the r/books page. So the useful skills from the shortcuts are Navigate to forums, Sort posts by hotness
</think>
<selected-skills>
id: 1; name: Navigate to forums
id: 3; name: Sort posts by hotness
</selected-skills>

Notes:
1. Some skills might not be consistent with the current task but it is still useful to refer to, e.g. write a post to express happiness is useful in a task to write a post to express sadness.
"""

# process the prompt

# add \\ at the end of each line
prompt = prompt.replace("\n", "\n\\\\")
prompt = prompt.replace("<", "$<$")
prompt = prompt.replace(">", "$>$")
print(prompt)
# save the prompt
with open("prompt.txt", "w") as f:
    f.write(prompt)