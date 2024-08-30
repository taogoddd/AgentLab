from agentlab.agents.ap_agent.ap_agent_prompt import AutoGenPolicyAgentPromptFlags
from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.chat_api import OpenAIChatModelArgs, AzureOpenAIChatModelArgs

flags = AutoGenPolicyAgentPromptFlags(
        obs=dp.ObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_focused_element=True,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=False,
            use_action_history=True,
            use_think_history=True,
            use_diff=False,
            html_type="pruned_html",
            use_screenshot=True,
            use_som=False,
            extract_visible_tag=True,
            extract_clickable_tag=True,
            extract_coords="False",
            filter_visible_elements_only=False,
        ),
        action=dp.ActionFlags(
            multi_actions=False,
            # change action space here
            action_set="wa_bid+reddit",
            long_description=True,
            individual_examples=True,
        ),
        use_plan=False,
        use_criticise=False,
        use_thinking=True,
        use_memory=False,
        use_concrete_example=True,
        use_abstract_example=True,
        use_hints=True,
        enable_chat=False,
        max_prompt_tokens=None,
        be_cautious=False,
        extra_instructions="", # add extra instructions here
    )

chat_model_args = AzureOpenAIChatModelArgs(
        model_name="azureopenai/gpt-4o-2024-05-13",
        max_total_tokens=128_000,
        max_input_tokens=40_000,  # make sure we don't bust budget
        max_new_tokens=4000,  # I think this model has very small default value if we don't set max_new_tokens
        vision_support=True,
    ),