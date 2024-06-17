import { OpenAI } from 'openai'; // Adjust the import according to your setup

interface Message {
    role: string;
    content: string;
}

const config = {
    apiKey: process.env.OPENAI_API_KEY
};

const client = new OpenAI(config);

export const openAiClient = async (messages: Message[]): Promise<any> => {
    if (!process.env.OPENAI_MODEL) {
        throw new Error("OPENAI_MODEL environment variable is not defined");
    }

    const chatMessages: any = messages.map((message) => ({
        role: message.role,
        content: message.content,
    }));

    try {
        const response = await client.chat.completions.create({
            model: process.env.OPENAI_MODEL,
            messages: chatMessages,
        });

        if (response.choices && response.choices[0] && response.choices[0].message) {
            return response.choices[0].message.content;
        } else {
            throw new Error("Invalid response from OpenAI API");
        }
    } catch (error) {
        console.error('Error making OpenAI API request:', error);
        throw error;
    }
};
