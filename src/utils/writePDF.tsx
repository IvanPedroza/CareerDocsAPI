import React from 'react';
import { Page, Text, View, Document, StyleSheet, Font, renderToBuffer } from '@react-pdf/renderer';
import info from "./info.json";

// Register the variable font
Font.register({
  family: 'EB Garamond',
  fonts: [
    { src: 'src/public/fonts/EBGaramond-Regular.ttf', fontWeight: 400 },
    { src: 'src/public/fonts/EBGaramond-Bold.ttf', fontWeight: 700 },
    { src: 'src/public/fonts/EBGaramond-Italic.ttf', fontStyle: 'italic' },
  ]
});

// Constants for styling
const pageMargin = 35;
const footerHeight = 50;
const fontColorPrimary = '#000';

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

export const splitResumeSummary = (resumeSummary: string): [string, string] => {
  // Define a regular expression to match "Previous work experience" with some tolerance
  const regex = /previous.*work.*experience/i;

  // Find the index of the match
  const match = resumeSummary.match(regex);
  if (!match) {
      // If no match is found, return null
      return ["", ""];
  }

  // Split the resume summary at the match index
  const splitIndex: number = match.index!;
  const contactInfo: string = resumeSummary.substring(0, splitIndex).trim();
  const workExperience: string = resumeSummary.substring(splitIndex).trim();

  return [contactInfo, workExperience];
}

export type ContactInfo = {
  applicantName: string;
  applicantTitle: string;
  applicantEmail: string;
  applicantResidence: string;
  applicantPortfolioWebsite: string;
  applicantGithub: string;
  applicantPhone: string;
}

const contactFields = {
  name: /"Applicant name":\s*"([^"]*)"/i,
  title: /"Applicant title":\s*"([^"]*)"/i,
  email: /"Applicant email":\s*"([^"]*)"/i,
  residence: /"Applicant residence":\s*"([^"]*)"/i,
  portfolio: /"Applicant portfolio website":\s*"([^"]*)"/i,
  github: /"Applicant github":\s*"([^"]*)"/i,
  phone: /"Applicant phone":\s*"([^"]*)"/i,
};

export function extractContactInfo(contactInfo: string): ContactInfo {


  const extractField = (pattern: RegExp): string => {
    const match = contactInfo.match(pattern);
    return match && match[1] ? match[1] : "";
  };

  return {
    applicantName: extractField(contactFields.name),
    applicantTitle: extractField(contactFields.title),
    applicantEmail: extractField(contactFields.email),
    applicantResidence: extractField(contactFields.residence),
    applicantPortfolioWebsite: extractField(contactFields.portfolio),
    applicantGithub: extractField(contactFields.github),
    applicantPhone: extractField(contactFields.phone),
  };
}


export const generatePDFDocument = (hook: string, body:string, closing:string): React.JSX.Element => (
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



export const writeCoverLetterPDF = (final: string, contactInfo: ContactInfo): React.JSX.Element => (
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
