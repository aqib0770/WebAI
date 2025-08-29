import { tool } from "@langchain/core/tools";
import { chromium } from "playwright";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatGroq } from "@langchain/groq";
import "dotenv/config";
import { JSDOM } from "jsdom";
import readline from "readline";

let browser;
let page;
let innerHTML;

function cleanHTML(html) {
  const dom = new JSDOM(html);
  const document = dom.window.document;
  document
    .querySelectorAll("[class]")
    .forEach((el) => el.removeAttribute("class"));
  document
    .querySelectorAll("[style]")
    .forEach((el) => el.removeAttribute("style"));
  document
    .querySelectorAll(
      "script, style, meta, link, noscript, svg, path, iframe, img, canvas"
    )
    .forEach((el) => el.remove());

  document.querySelectorAll("div").forEach((div) => {
    const hasSemantic = div.querySelector(
      "a, button, input, select, textarea, label, form, header, footer, main, nav, section, article"
    );
    if (!hasSemantic && div.textContent.trim() === "") {
      div.remove();
    }
  });
  return document.body.innerHTML;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export const openBrowserTool = tool(
  async () => {
    if (!browser) {
      browser = await chromium.launch({ headless: false });
      page = await browser.newPage();
      page.setDefaultTimeout(15000);
    }
    await sleep(2000); // Wait for 2 seconds to ensure the browser is fully ready
    return "✅ Browser opened and ready.";
  },
  {
    name: "open_browser",
    description: "Opens a browser (singleton). MUST be called first.",
    schema: z.object({}),
  }
);

export const navigateTool = tool(
  async ({ url }) => {
    // console.log("Navigating to:", url);
    if (!page) throw new Error("Browser not opened. Call open_browser first.");
    await page.goto(url, { timeout: 20000, waitUntil: "networkidle" });
    const rawHTML = await page.innerHTML("body");
    innerHTML = cleanHTML(rawHTML);
    // console.log("Navigation complete.");
    await sleep(1000);
    return { status: "✅ Navigated", url, innerHTML };
  },
  {
    name: "navigate",
    description: "Navigates to a URL and updates DOM state.",
    schema: z.object({
      url: z.string().url().describe("The URL to open"),
    }),
  }
);

export const findInputLabelsTool = tool(
  async () => {
    // console.log("Finding input labels");
    if (!page) throw new Error("Browser not opened. Call open_browser first.");
    const html = await page.innerHTML("body");
    const dom = new JSDOM(html);
    const labels = Array.from(
      dom.window.document.querySelectorAll("label")
    ).map((label) => ({
      for: label.getAttribute("for"),
      text: label.textContent.trim(),
    }));
    // console.log("Found labels:", labels);
    await sleep(500);
    return { labels };
  },
  {
    name: "find_input_labels",
    description: "Extracts all visible input labels from the current page.",
    schema: z.object({}),
  }
);

export const findElementTool = tool(
  async ({ text }) => {
    // console.log("Finding element with text:", text);
    if (!page) throw new Error("Browser not opened. Call open_browser first.");
    const locator = page.locator(`text=${text}`);
    const count = await locator.count();
    if (count === 0)
      throw new Error(`❌ Element with text "${text}" not found.`);
    // console.log(`Found ${count} elements with text "${text}".`);
    await sleep(500);
    return { selector: `text=${text}`, found: true };
  },
  {
    name: "find_element",
    description:
      "Finds an element strictly by visible text and returns its selector.",
    schema: z.object({
      text: z.string().describe("The exact visible text of the element"),
    }),
  }
);

export const clickElementTool = tool(
  async ({ selector }) => {
    // console.log("Clicking element with selector:", selector);
    if (!page) throw new Error("Browser not opened. Call open_browser first.");
    await page.click(selector);
    const rawHTML = await page.innerHTML("body");
    innerHTML = cleanHTML(rawHTML);
    // console.log("Click action complete.");
    await sleep(1000);
    return { status: `✅ Clicked element`, selector, innerHTML };
  },
  {
    name: "click_element",
    description: "Clicks an element using its selector.",
    schema: z.object({
      selector: z.string().describe("The selector of the element to click"),
    }),
  }
);

export const submitButtonTool = tool(
  async ({ buttonName }) => {
    // console.log("Clicking button with name:", buttonName);
    if (!page) throw new Error("Browser not opened. Call open_browser first.");
    const button = page.getByRole("button", { name: buttonName });
    if ((await button.count()) === 0) {
      throw new Error(`❌ Button "${buttonName}" not found.`);
    }
    await button.click();
    const rawHTML = await page.innerHTML("body");
    innerHTML = cleanHTML(rawHTML);
    // console.log(`Button "${buttonName}" clicked. Updated innerHTML:`);
    await sleep(1000);
    return { status: `✅ Clicked button`, button: buttonName, innerHTML };
  },
  {
    name: "submit_button",
    description: "Clicks a button by its visible name.",
    schema: z.object({
      buttonName: z.string().describe("The exact text of the button"),
    }),
  }
);

export const fillInputTool = tool(
  async ({ label, value }) => {
    // console.log(`Filling input "${label}" with value:`, value);
    if (!page) throw new Error("Browser not opened. Call open_browser first.");
    await page.getByLabel(label, { exact: true }).fill(value);
    console.log(`Filled input "${label}".`);
    await sleep(500);
    return { status: `✅ Filled input`, label, value };
  },
  {
    name: "fill_input",
    description: "Fills an input field by its label text.",
    schema: z.object({
      label: z.string().describe("The label of the input field"),
      value: z.string().describe("The value to type"),
    }),
  }
);

export const closeBrowserTool = tool(
  async () => {
    // console.log("Closing browser");
    if (browser) {
      await browser.close();
      browser = undefined;
      page = undefined;
      return "✅ Browser closed.";
    }
    return "⚠️ Browser was not open.";
  },
  {
    name: "close_browser",
    description:
      "Closes the browser. Must only be called after task completion.",
    schema: z.object({}),
  }
);

const model = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "meta-llama/llama-4-scout-17b-16e-instruct",
});

const agent = createReactAgent({
  tools: [
    openBrowserTool,
    navigateTool,
    findElementTool,
    clickElementTool,
    fillInputTool,
    submitButtonTool,
    findInputLabelsTool,
    closeBrowserTool,
  ],
  llm: model,
});

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  rl.question("Enter your web automation task: ", async (task) => {
    const result = await agent.stream({
      messages: [
        {
          role: "system",
          content: `
          You are a strict web automation agent. Follow these exact rules:

          1. Always call tools in the correct sequence:
            - open_browser → navigate → find_element → click_element/fill_input → submit_button → close_browser.

          2. After every navigation or click, always use the latest DOM (innerHTML).

          3. Do not guess element names. Always use find_element or find_input_labels first.

          4. Never call a tool if it is not required.

          5. IMPORTANT: When the task is finished:
            - First, output a short natural language summary of the steps you took and the outcome.
            - Then, in the NEXT step, call the close_browser tool.
            - Do NOT combine the summary and the tool call in the same response.

          6. close_browser must always be the very last action. After calling close_browser, STOP. Do not take any further actions.

          7. If an element is missing, stop with an error. Do not hallucinate.

          8. You must NEVER try to parse or analyze raw HTML yourself.

          9. If a selector is needed for an element, ALWAYS use the tool 'element_finder'.

          You are not allowed to perform reasoning without a tool call, except in the final summary.
          `,
        },
        {
          role: "user",
          content: task,
        },
      ],
    });
    for await (const chunk of result) {
      if (chunk.agent && chunk.agent.messages.length > 0) {
        const message = chunk.agent.messages[0];

        if (message.tool_calls && message.tool_calls.length > 0) {
          for (const toolCall of message.tool_calls) {
            console.log(`🤖 Agent wants to run tool: **${toolCall.name}**`);
            console.log(`   Arguments: ${JSON.stringify(toolCall.args)}`);
          }
        } else if (message.content) {
          console.log(`✅ Final Answer: ${message.content}`);
        }
      } else if (chunk.tools && chunk.tools.messages.length > 0) {
        for (const toolMessage of chunk.tools.messages) {
          console.log(`🛠️ Tool **${toolMessage.name}** responded:`);
          try {
            const content = JSON.parse(toolMessage.content);
            console.log(content);
          } catch (e) {
            console.log(toolMessage.content);
          }
        }
      }
    }
    rl.close();
  });
}

main();
