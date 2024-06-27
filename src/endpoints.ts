import { z } from 'zod';
import { Routing, defaultEndpointsFactory, createResultHandler, ez, EndpointsFactory } from 'express-zod-api';
import { GPT } from './models/OpenAIClient';
import { StringifyPdf } from './utils/DocumentLoader';
import { writeCoverLetterPDF, splitResumeSummary, extractContactInfo, ContactInfo } from './utils/writePDF';
import { generateJobListingPrompt } from './prompts/jobListingPrompt';
import { generateResumePrompt } from './prompts/resumePrompt';
import { generateHookPrompt } from './prompts/hookPrompt';
import { generateBodyPrompt } from './prompts/bodyPrompt';
import { generateReviewPrompt } from './prompts/reviewPrompt';
import { generateConclusionPrompt } from './prompts/conclusionPrompt';
import { generateFinalPrompt } from './prompts/truthPrompt';
import ReactPDF, { renderToBuffer } from '@react-pdf/renderer';
import { fetchListing } from './utils/jobs';
import { Gemini } from './models/Gemini';
import { loggerFactory } from './utils/logger';

const parentLogger = loggerFactory();
const createLogger = () => parentLogger.child({ correlationId: crypto.randomUUID() });

// --- Persona Definitions ---
const talentAcquisitionExpertPersona: string = `You are an expert in talent acquisition \
and workforce optimization with 20 years of experience summarizing job descriptions.`;

const coverLetterWriterPersona: string = `You are an expert cover letter writer with a \
comprehensive understanding of Applicant Tracking Systems (ATS) and keyword optimization.`;



const resume: string = StringifyPdf("src/public/Ivan Pedroza Resume.pdf");


const generateLetter = async (url: string, model: string) => {

  const jobListing = await fetchListing(url);
  const jobListingPrompt = generateJobListingPrompt(jobListing);

  const Model = model === 'GPT' ? GPT : Gemini;

  const [jobSummary, resumeSummary] = await Promise.all([
    Model([
      { role: "system", content: talentAcquisitionExpertPersona },
      { role: "user", content: jobListingPrompt }
    ]),
    Model([
      { role: "system", content: talentAcquisitionExpertPersona },
      { role: "user", content: generateResumePrompt(resume) }
    ])
  ]);

  const resumeSplit = splitResumeSummary(resumeSummary);
  let [contactInfo, workExperience] = resumeSplit;

  const hook = await Model([
    { role: "system", content: coverLetterWriterPersona },
    { role: "user", content: generateHookPrompt(workExperience, jobSummary) }
  ]);

  const body = await Model([
    { role: "system", content: coverLetterWriterPersona },
    { role: "user", content: generateBodyPrompt(workExperience, jobSummary, hook) }
  ]);

  const revised_body = await Model([
    { role: "system", content: coverLetterWriterPersona },
    { role: "user", content: generateReviewPrompt(workExperience, jobSummary, body) }
  ]);

  const conclusion = await Model([
    { role: "system", content: coverLetterWriterPersona },
    { role: "user", content: generateConclusionPrompt(hook, revised_body) }
  ]);

  const final = await Model([
    { role: "system", content: coverLetterWriterPersona },
    { role: "user", content: generateFinalPrompt(workExperience, hook, revised_body, conclusion) }
  ]);

  const contactValues: ContactInfo = extractContactInfo(contactInfo);

  return writeCoverLetterPDF(final, contactValues);
};


// --------------------------------------------------------- ENDPOINTS ------------------------------------------------------

const pdfEndpoint = new EndpointsFactory(
  createResultHandler({
    getPositiveResponse: () => ({
      schema: z.instanceof(Buffer),
      mimeType: "application/pdf",
    }),
    getNegativeResponse: () => ({ schema: z.string(), mimeType: "text/plain" }),
    handler: ({ response, error, output }) => {
      if (error) {
        response.status(500).send(error.message);
        return;
      }
      if (output && output.file) {
        const pdfBuffer = output.file as Buffer;
        response
          .status(200)
          .type('application/pdf')
          .send(pdfBuffer);
      } else {
        response.status(400).send("No file found");
      }
    },
  })
);

const test = defaultEndpointsFactory.build({
  shortDescription: "fetches job listing",
  description: 'retrieves text from job listing',
  method: 'get',
  input: z.object({}),
  output: z.object({ text: z.string() }),
  handler: async () => {
    const logger = createLogger().child({ endpoint: "/test" });
    logger.error(`Endpoint called: /test, data: ${JSON.stringify({"test": "test"})}`);
    return { text: "Hello World!" };
  },
});

const coverLetterEndpoint = pdfEndpoint.build({
  shortDescription: "Fetches cover letter files",
  description: 'Retrieves most up-to-date general cover letter',
  method: 'post',
  input: z.object({
    jobUrl: z.string(),
    model: z.enum(['gpt', 'gemini'])
  }),
  output: z.object({ file: z.instanceof(Buffer) }),
  handler: async ({ input }): Promise<{ file: Buffer }> => {

    const { jobUrl, model } = input;
    const logger = createLogger().child({ endpoint: "/cover" });
    logger.info("Endpoint called: /cover", { data: JSON.stringify(input) });
    const pdfDocument = await generateLetter(jobUrl, model);

    if (!pdfDocument) {
      logger.error(`Failed to generate PDF document with the following input:  ${JSON.stringify(input)}`);
      throw new Error('Failed to generate PDF document. Please verify the input and try again.');
    }

    const file = await renderToBuffer(pdfDocument);
    return { file };
  },
});

// --------------------------------------------------------- ROUTING ------------------------------------------------------

export const appRouter: Routing = {
  test: test,
  cover: coverLetterEndpoint,
};
