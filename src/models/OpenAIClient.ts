import { OpenAI } from 'openai'; 
import { wrapOpenAI } from 'langsmith/wrappers'; 
import { loggerFactory } from '../utils/logger';
import { log } from 'winston';

interface Message {
    role: string;
    content: string;
}

const parentLogger = loggerFactory();
const createLogger = () => parentLogger.child({ correlationId: crypto.randomUUID() });
const logger = createLogger().child({ module: "/OpenAIClient" });

const apiKey = process.env.OPENAI_API_KEY;

if (!apiKey) {
    logger.error(`OpenAI API key was not provided.  Key: ${apiKey}`);
    throw new Error(`Valid OpenAI API key must be provided. Unable to send request with key: ${apiKey}`)
}

const client = wrapOpenAI(new OpenAI({ apiKey: apiKey }));



export const GPT = async (messages: Message[]): Promise<string> => {

    const chatMessages: any = messages.map((message) => ({
        role: message.role,
        content: message.content,
    }));

    try {
        const response = await client.chat.completions.create({
            model: process.env.OPENAI_MODEL!,
            messages: chatMessages,
        });

        if (response.choices && response.choices[0] && response.choices[0].message) {
            return response.choices[0].message.content!;
        } else {
            logger.error(`Invalid response from OpenAI API to prompt: ${prompt}. Response: ${response}`);
            throw new Error(`Invalid response from OpenAI API to prompt: ${prompt}`);
        }
    } catch (error: any) {

        const errorContext = {
            error: error.message,
            stack: error.stack,
            prompt,
        };
        logger.error('Error making Open AI API request', errorContext);

        throw new Error(`Failed to generate content with Gemini API. Context: ${JSON.stringify(errorContext)}`);
    }
};
