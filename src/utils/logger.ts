import { createLogger, format, transports } from 'winston';
import { SeqTransport } from '@datalust/winston-seq';


export const loggerFactory = () => createLogger({
  format: format.combine( 
    format.errors({ stack: true }),
    format.json(),
  ),
  defaultMeta: { application: 'career-docs-api' },
  transports: [
    [
      new transports.Console({
        format: format.simple(),
      })
    ],

    process.env.NODE_ENV === "production" ? [
      new SeqTransport({
        serverUrl: process.env.SEQ_SERVER,
        apiKey: process.env.SEQ_API_KEY,
        onError: (e => { console.error(e) }),
        handleExceptions: true,
        handleRejections: true,
      })
    ] : []

  ].flat(),
});