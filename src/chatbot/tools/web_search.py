"""
Web search capability for the AI agent using Tavily.

This module is built with external provider flexibility in
mind so they can easily be switched if necessary.
"""
import json
from typing import Any

import requests
from loguru import logger

from config import AppConfig

# ------------------------------------------------------------------
# General- purpose utils


def web_search_tool_available(
        provider_config: dict = AppConfig.TAVILY_CONFIG
) -> bool:
    """
    Determine whether the web search tool is available for the agent.

    Parameters
    ----------
        provider_config: dict
            A dictionary containing the external provider's
            URL and API key. Required keys: "url", "api key".

    Returns
    -------
        bool
            True if the provider's parameters are correctly set and
            the tool can be used, False otherwise.

    Examples
    --------
    >>> result = web_search_tool_available({
            "url": "https://api.tavily.com/search",
            "api key": "example-api-key"
        })
    >>> print(result)
    True
    """
    if not isinstance(provider_config, dict) or not provider_config:
        return False

    # Test URL and API key
    for var in ["url", "api key"]:
        if (
            var not in provider_config
            or (not isinstance(var, str))
            or len(var) == 0
        ):
            return False

    return True


# ------------------------------------------------------------------
# Tavily

class TavilyClient:
    """Class used for handling Tavily searches."""

    def __init__(
            self,
            config: dict = AppConfig.TAVILY_CONFIG
    ) -> None:
        """
        Init the TavilyClient class instance based on env vars.

        Currently assumes that only the web search endpoint is being used,
        adapt this method if using other Tavily functionalities
        (e.g., extract, crawl, map)
        """
        self.url = config["url"]
        self.api_key = config["api key"]
        self._build_headers()

    def _build_headers(self) -> None:
        """Build the headers for connecting to the Tavily API."""
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def search(
            self,
            query: str,
            include_answer: bool = True,
            simplify_response_for_agent: bool = False,
            **kwargs: Any
    ) -> dict | str:
        """
        Perform a web search using Tavily.

        Includes optional kwargs that, if not provided or None, aren't
        sent in the API call. This is done on purpose instead of defining
        optional parameters by name for easier maintainability in case the
        endpoint definition ever changes on the provider's side.

        Docs:
        https://docs.tavily.com/documentation/api-reference/endpoint/search

        Parameters
        ----------
            query: str
                The search query.

            include_answer: bool = True
                If True, includes an LLM-generated answer summarizing the
                results for the query.
                It's recommended to set this value to True if using within
                an agent tool, in order to return a simple, short answer.

            simplify_response_for_agent: bool = False
                If True, returns a string with the search results.
                Otherwise returns a dict, which is the original format of
                the response content.

            **kwargs
                Optional kwargs to send to Tavily to further customize
                the obtained response.

        Returns
        -------
            str
                If simplify_response = True
            dict
                If simplify_response = False. i.e., the original response
                returned by the API.

        Examples
        --------
        >>> client = TavilyClient()

        Successful call without simplifying response:

        >>> response = client.search(
            query="Tesla NYSE stock price today"
        )
        >>> print(response)
        {
        "query": "Tesla NYSE stock price today",
        "follow_up_questions": null,
        "answer": "Tesla NYSE stock price today is $393.45, ...",
        "images": [],
        "results": [
            {
            "url": "https://www.tradingview.com/symbols/NASDAQ-TSLA",
            "title": "TSLA Stock Price — Tesla Chart - TradingView",
            "content": "The current price of TSLA is 393.45 USD ...",
            "score": 0.78194773,
            "raw_content": null
            },
            ...
            }
        ],
        "response_time": 1.17,
        "request_id": "8f850ab2-6a8e-473a-8385-9e21f23e2968"
        }

        Successful call with simplified response:

        >>> response = client.search(
            query="Tesla NYSE stock price today",
            simplify_response_for_agent=True
        )
        >>> print(response)
        "Tesla NYSE stock price today is $393.45, ..."


        """
        # LLM answer is required for simplifying LLM response
        if simplify_response_for_agent:
            include_answer = True

        # Solve params and send request
        params = {
            "query": query,
            "include_answer": include_answer
        }
        if kwargs:
            params.update({
                k: v for k, v in kwargs.items()
                if v is not None and k not in params.keys()
            })

        response = requests.post(
            self.url,
            headers=self.headers,
            data=json.dumps(params)
        )

        # Parse response for return value
        if response.status_code == 200:
            response_body = response.json()
            if simplify_response_for_agent:
                return response_body["answer"]
            return response_body
        else:
            logger.error(
                "Unsuccessful Tavily API call. "
                f"Status code: {response.status_code}, "
                f"Content: {response.content}"
            )
            if simplify_response_for_agent:
                return (
                    "Error: Web search tool returned status "
                    f"{response.status_code} and is not available."
                )
            return response.content
