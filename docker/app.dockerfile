FROM node:22-slim

WORKDIR /app

COPY package*.json /app
COPY tsconfig*.json /app
COPY ./src /app/src

RUN npm ci && npm run build



EXPOSE 3000 5341

CMD ["npm", "run", "start:prod"]


