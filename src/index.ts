import dotenv from 'dotenv';
dotenv.config();
import { appRouter } from "./endpoints";
import express, { Request, Response, NextFunction } from "express";
import { attachRouting, AppConfig } from "express-zod-api";
import swaggerJsdoc from 'swagger-jsdoc';
import swaggerUi from 'swagger-ui-express';


const app = express();
const PORT = 3000;

const config: AppConfig<string> = {
    app,
    cors: true,
    logger: {level: 'debug', color: true},
};


const options = {
    definition: {
        openapi: "3.0.0",
        info: {
            title: "Career Documents API",
            version: "1.0.0",
            description: "A simple Express Library API to serve resume and cover letter files.",
        },
        servers: [{
            url: `http://35.93.146.27:${PORT}`
        }],
    },
    apis: ["./API_Documentation.yaml"],
};

const swaggerSpec = swaggerJsdoc(options);

app.use("/api-docs", swaggerUi.serve, swaggerUi.setup(swaggerSpec));
app.use(express.json());

attachRouting(config, appRouter);

// Error handling middleware
app.use((err: any, req: Request, res: Response, next: NextFunction) => {
    console.error(err.stack);
    res.status(500).json({ message: 'An unexpected error occurred! Here are the logs: ', error: err.message });
});


const server = app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

const shutdown = () => {
    console.log('Received shutdown signal, shutting down gracefully...');
    server.close(() => {
        console.log('Closed out remaining connections.');
        process.exit();
    });

    setTimeout(() => {
        console.error('Forcing shutdown as connections are taking too long to close.');
        process.exit(1);
    }, 10000);
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    shutdown();
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
    shutdown();
});