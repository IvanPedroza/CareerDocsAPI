import { JwtPayload } from 'jsonwebtoken';

declare module 'express' {
  interface Request{
    user?: string | JwtPayload;
    jwtToken?: string ;
  }
}
