import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "体積ハカルくん｜Colab試作版",
  description: "iPhoneで撮影した動画から対象物の体積を推定するGoogle Colab試作版",
};

export default function VolumeLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
