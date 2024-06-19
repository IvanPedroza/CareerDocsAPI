import React from 'react';
import fs from 'fs';
import path from 'path';
import { Page, Text, View, Document, StyleSheet, Font, renderToBuffer } from '@react-pdf/renderer';
import info from "./info.json";

// Register the variable font
Font.register({
  family: 'EB Garamond',
  fonts: [
    { src: 'public/fonts/EBGaramond-Regular.ttf', fontWeight: 400 },
    { src: 'public/fonts/EBGaramond-Bold.ttf', fontWeight: 700 },
    { src: 'public/fonts/EBGaramond-Italic.ttf', fontStyle: 'italic' },
  ]
});

// Constants for styling
const pageMargin = 35;
const footerHeight = 50;
const fontColorPrimary = '#000';
const fontColorSecondary = '#666';

// Create styles
const styles = StyleSheet.create({
  page: {
    fontSize: 10,
    padding: pageMargin,
    paddingBottom: footerHeight,
    fontFamily: 'EB Garamond',
    color: fontColorPrimary
  },
  name: {
    fontSize: 25,
    fontWeight: 700,
    color: fontColorPrimary,
    paddingLeft: 0,
  },
  container: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
  },
  nameSection: {
    flex: 2,
    justifyContent: 'flex-start',
    paddingLeft: 0,
  },
  contactInfoSection: {
    flex: 1,
    justifyContent: 'flex-end',
    textAlign: 'right',
    paddingRight: 20,
  },
  greeting: {
    fontSize: 10,
    fontWeight: 700,
    color: fontColorPrimary,
    marginBottom: 10,
  },
  paragraph: {
    paddingBottom: 10,
  },
  bulletPointsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingBottom: 0,
  },
  bulletPointsSection: {
    flex: 1,
    flexDirection: 'column',
    padding: 10,
  },
  bulletPoint: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 5,
  },
  bullet: {
    width: 3,
    height: 3,
    marginRight: 5,
    marginTop: 3,
    borderRadius: 1.5,
    backgroundColor: fontColorPrimary,
  },
  bulletText: {
    flex: 1,
    fontSize: 10,
    color: fontColorPrimary,
  },
});

const Gap = () => (
  <View style={{ marginTop: 7, marginBottom: 7 }}></View>
)

const Paragraph = ({ children }: { children: string | string[] }) => (
  <View style={styles.paragraph}>
    <Text>{children}</Text>
  </View>
)

const BulletPoint = ({ text }: { text: string }) => (
  <View style={styles.bulletPoint}>
    <View style={styles.bullet}></View>
    <Text style={styles.bulletText}>{text}</Text>
  </View>
);

function getCurrentFormattedDate(): string {
  const months = [
      "January", "February", "March", "April", "May", "June", 
      "July", "August", "September", "October", "November", "December"
  ];

  const date = new Date();
  const day = date.getDate();
  const month = months[date.getMonth()];
  const year = date.getFullYear();

  return `${day} ${month} ${year}`;
}

export function splitResumeSummary(resumeSummary: string): [string, string] | null {
  // Define a regular expression to match "Previous work experience" with some tolerance
  const regex = /previous.*work.*experience/i;

  // Find the index of the match
  const match = resumeSummary.match(regex);
  if (!match) {
      // If no match is found, return null
      return null;
  }

  // Split the resume summary at the match index
  const splitIndex = match.index!;
  const contactInfo = resumeSummary.substring(0, splitIndex).trim();
  const workExperience = resumeSummary.substring(splitIndex).trim();

  return [contactInfo, workExperience];
}

export interface ExtractedContactInfo {
  applicantName: string;
  applicantTitle: string;
  applicantEmail: string;
  applicantResidence: string;
  applicantPortfolioWebsite: string;
  applicantGithub: string;
  applicantPhone: string;
}

export function extractContactInfo(contactInfo: string): ExtractedContactInfo {
  const fields = {
    applicantName: /"Applicant name":\s*"([^"]*)"/i,
    applicantTitle: /"Applicant title":\s*"([^"]*)"/i,
    applicantEmail: /"Applicant email":\s*"([^"]*)"/i,
    applicantResidence: /"Applicant residence":\s*"([^"]*)"/i,
    applicantPortfolioWebsite: /"Applicant portfolio website":\s*"([^"]*)"/i,
    applicantGithub: /"Applicant github":\s*"([^"]*)"/i,
    applicantPhone: /"Applicant phone":\s*"([^"]*)"/i,
  };

  const extractField = (pattern: RegExp): string => {
    const match = contactInfo.match(pattern);
    return match && match[1] ? match[1] : "";
  };

  return {
    applicantName: extractField(fields.applicantName),
    applicantTitle: extractField(fields.applicantTitle),
    applicantEmail: extractField(fields.applicantEmail),
    applicantResidence: extractField(fields.applicantResidence),
    applicantPortfolioWebsite: extractField(fields.applicantPortfolioWebsite),
    applicantGithub: extractField(fields.applicantGithub),
    applicantPhone: extractField(fields.applicantPhone),
  };
}


export const generatePDFDocument = ({hook, body, closing}: {hook: string, body: string, closing: string}) => (
  <Document>
    <Page size="A4" style={styles.page}>
      <View style={styles.container}>
        <View style={styles.nameSection}>
          <Text style={styles.name}>Ivan Pedroza</Text>
        </View>
        <View style={styles.contactInfoSection}>
          <Text>Seattle, WA 98107</Text>
          <Text>204 5394-2345</Text>
          <Text>ivan.k.pedroza@gmail.com</Text>
        </View>
      </View>
      <Gap />
      <Text>{getCurrentFormattedDate()}</Text>
      <Gap />
      <Text>88 Colin P. Kelly Jr. Street</Text>
      <Text>San Francisco, CA 94107</Text>
      <Gap />
      <Text>To Whom It May Concern</Text>
      <Gap />
      <Paragraph>{hook}</Paragraph>
      <Paragraph>{body}</Paragraph>
      {/* <Paragraph>{info.Paragraph3}</Paragraph>
      <Paragraph>{info.Paragraph4}</Paragraph>
      <Paragraph>{info.skills_intro}</Paragraph> */}

      <View style={styles.bulletPointsContainer}>
        <View style={styles.bulletPointsSection}>
          <BulletPoint text={info.want1} />
          <BulletPoint text={info.want2} />
          <BulletPoint text={info.want3} />
        </View>
        <View style={styles.bulletPointsSection}>
          <BulletPoint text={info.have1} />
          <BulletPoint text={info.have2} />
          <BulletPoint text={info.have3} />
        </View>
      </View>
      <Paragraph>{closing}</Paragraph>
      <Text>Sincerely,</Text>
      <Text>Ivan Pedroza</Text>
    </Page>
  </Document>
);



export const writeCoverLetterPDF = ({
  final,
  contactInfo,
}: {
  final: string;
  contactInfo: ExtractedContactInfo;
}) => (
  <Document>
    <Page size="A4" style={styles.page}>
      <View style={styles.container}>
        <View style={styles.nameSection}>
          <Text style={styles.name}>{contactInfo.applicantName}</Text>
        </View>
        <View style={styles.contactInfoSection}>
          <Text>{contactInfo.applicantResidence}</Text>
          <Text>{contactInfo.applicantPhone}</Text>
          <Text>{contactInfo.applicantEmail}</Text>
          <Text>{contactInfo.applicantPortfolioWebsite}</Text>
        </View>
      </View>
      <Gap />
      <Text>{getCurrentFormattedDate()}</Text>
      <Gap />
      <Text>88 Colin P. Kelly Jr. Street</Text>
      <Text>San Francisco, CA 94107</Text>
      <Gap />
      <Paragraph>{final}</Paragraph>
      <Text>Sincerely,</Text>
      <Text>{contactInfo.applicantName}</Text>
    </Page>
  </Document>
);
