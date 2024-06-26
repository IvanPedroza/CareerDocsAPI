import { GoogleGenerativeAI } from "@google/generative-ai";
import { loggerFactory } from "../utils/logger";
import crypto from "crypto";

interface Message {
    role: string;
    content: string;
}

const parentLogger = loggerFactory();
const createLogger = () => parentLogger.child({ correlationId: crypto.randomUUID() });
const logger = createLogger().child({ module: "/Gemini" });

const apiKey = process.env.GEMINI_API_KEY;
const modelName = process.env.GEMINI_MODEL || "gemini-1.5-flash";

if (!apiKey) {
    logger.error(`Gemini API key was not provided.  Key: ${apiKey} Model: ${modelName}`);
    throw new Error("Gemini API key was not provided.");
}

const genAI = new GoogleGenerativeAI(apiKey);
const model = genAI.getGenerativeModel({ model: modelName });

export const Gemini = async (messages: Message[]): Promise<string> => {
    const prompt: string = messages.map(message => `${message.role}: ${message.content}`).join('\n');

    try {
        const result = await model.generateContent(prompt);
        return result.response.text();
    } catch (error: any) {
        const errorContext = {
            error: error.message,
            stack: error.stack,
            prompt,
        };
        logger.error('Error making Gemini API request', errorContext);

        throw new Error(`Failed to generate content with Gemini API. Context: ${JSON.stringify(errorContext)}`);
    }
};