import { computed, ref } from 'vue'

const savedLocale = localStorage.getItem('climatenet-locale')
export const locale = ref(savedLocale === 'en' ? 'en' : 'zh')

export function setLocale(nextLocale) {
  locale.value = nextLocale === 'en' ? 'en' : 'zh'
  localStorage.setItem('climatenet-locale', locale.value)
}

export function useI18n() {
  const isZh = computed(() => locale.value === 'zh')
  const pick = (zh, en) => (isZh.value ? zh : en)
  return { locale, isZh, pick, setLocale }
}
